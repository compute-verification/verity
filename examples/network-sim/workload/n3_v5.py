"""N3Window_v5: `n3_v4`'s N3Window_v4 over outbox rows without overlap, and, with a template, its frames' free fields
fixed. It accepts exactly where v4 does on every window the tests cover, and each read hashes one B-byte row where v4's
hashes a 2B-byte one.

    N3Window_v5{FMT, N, G, SALTED, REC, R, TPL, FB}
        (layout, layout_domain: Digest, at: Array<DL>, record, record_domain: Digest, direction: Array<1>,
         roots, domains: Array<U, Digest>, [salt: Array<72, Byte>],
         frames: Array<N, Array<FB, Byte>>,
         descriptors: Array<n, Array<LW>>, descriptor_salts: Array<n, Salt>,
         descriptor_left, descriptor_right: Array<DL, Digest>,
         rows: Array<n, G, Raw>, row_salts: Array<n, G, Salt>,
         runs: Array<T, Array<R>>, fresh: Array<T>,
         run_units: Array<R, Array<UB>>, run_starts: Array<R, Array<DB>>,
         run_heads: Array<R, Raw>, run_head_salts: Array<R, Salt>, run_fill: Array<R, Array<S - 1, Digest>>,
         run_left, run_right: Array<R, Array<DB, Digest>>,
         head: Array<HL, Byte>, record_salt: Salt, record_path: Array<2, Digest>, [slot: Array<SB>]) -> Value<1>
        Raw = Array<B, Byte>, T = nG, S = min(T + 1, 2^DB)

v4's statics, inputs and checks, except the outbox's rows, how a read reaches its row, and the frames' placement:

- **The outbox** (`outbox_rows`): v3's bytes in rows of B, each byte in one row: row q holds bytes qB ... qB + B - 1
  for q <= k = max(1, ceil(bytes / B)), so row k, past the data, is zero; then zero rows up to a power of two,
  registered under `OUTBOX_LABEL` (`register_outbox`). A segment keeps its meaning: it reads bytes o ... o + n - 1 of
  the 2B bytes from qB, its window, which is rows q and q + 1 and v4's row q.
- **A read hashes one row.** Read t's witness row is row q + 1 of unit u, its hiding leaf at q + 1 (`N3Next`,
  `N3HidingLeaf`). A kind-1 read is in exactly one run (`N3RunsOne`). Each run carries two rows, (0, run_heads[r]) to
  begin with: a fresh member moves them to (the second, its row), a repeat keeps them (`N3Advance`), and a read's
  window is its run's two rows after it, zero for kind 0 (`N3Gate`). Run r's slots open at run_starts[r],
  run_starts[r] + 1, ... in unit run_units[r]'s tree: slot 0 the head's leaf (`N3HidingLeaf` at the start), then its
  fresh leaves in order (v4's compaction network, each stage only where the next reads: `compaction`), then
  `run_fill`. So fresh member j's row is row start + j + 1, its q, which its leaf binds, is start + j, and its window
  rows start + j and start + j + 1. A repeat's leaf is its run's last member's (`N3Carry`), so it has that member's q
  and window. The head and the fresh leaves fit the unit's tree (`N3RunFits` with one slot more).
- **A template** (TPL, `template`): () reads each frame's placement from its descriptor, as v4 does. Otherwise
  (mode, p, s, prefix hex, suffix hex) fixes it for every frame of the window: the frame is prefix XOR text << 8p XOR
  suffix << 8s, the text the stream's first min(r, B) bytes (RAW) or their base64 (BASE64), r the descriptor's
  (`N3FrameFixed`, `N3PlaceFixed`). The descriptor's mode, p, s, prefix and suffix are not read, so a layout for such
  windows needs only r and the segments (P = Q = 1).
- **Frames wider than rows** (FB, B unless given): with a template, a frame may be FB > B bytes, its text from a
  stream of B (base64 of it cut at `b64_chars`); the salt and N2's frames root are over frames of FB. Rows then hold
  what a frame carries rather than what it could: `prepare(setup, RB)` opens the outboxes in rows of RB and `resplit`s
  each descriptor's segments onto them (a frame's data at most RB bytes).

v5's runs are v4's (`n3_v4.partition`, `runs_of`), and so is R. Its witness rows are the halves of v3's (`convert`):
read t's is the second half of v3's row t, and a run's head the first half of its first member's.

    python -m workload.n3_v5 circuit-check --out FILE [--window]   circuit-check on v5's new parts [and the demo's window]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, add, const_bits
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.lms import LmsEqual

from . import n3_unit as U
from . import n3_v4 as V4
from .n3_unit import BIT, BYTE, DIGEST, SALT_T

ZD = bytes(64)
NO_SALT = bytes(hm96.SHA512.salt_bytes)
OUTBOX_LABEL = b"verity/netsim/n3-outbox/v2/"


# --- the outbox -------------------------------------------------------------------------------------------------------

def outbox_rows(data: bytes, B: int) -> list[bytes]:
    """The outbox's rows of B bytes: row q holds bytes qB ... qB + B - 1, zero past the end, for q <= k = max(1,
    ceil(len / B)); then zero rows up to a power of two."""
    k = max(1, -(-len(data) // B))
    rows = [data[q * B:(q + 1) * B].ljust(B, b"\0") for q in range(k + 1)]
    return rows + [bytes(B)] * ((1 << k.bit_length()) - (k + 1))


def outbox_binding(program: bytes, unit: str) -> bytes:
    return hashlib.sha512(OUTBOX_LABEL + program + unit.encode()).digest()


def register_outbox(program: bytes, unit: str, data: bytes, B: int, seed: bytes) -> U.Tree:
    return U.register(outbox_rows(data, B), 8 * B, outbox_binding(program, unit), seed + b"/outbox/" + unit.encode())


@dataclass
class Prepared:
    """A setup's units with their outboxes registered as v5's rows, and its layout in the format of their depths."""

    units: U.UnitTable
    layout: U.Layout


def resplit(d: U.Descriptor, B: int, RB: int) -> U.Descriptor:
    """A descriptor of rows of B bytes on rows of RB: each segment's bytes in pieces of at most RB from their RB-byte
    row, only r and the segments kept (a template places the frame); refuses a frame whose stream doesn't fit RB."""
    if d.r > RB:
        raise ValueError(f"a frame of {d.r} data bytes doesn't fit rows of {RB}")
    segs = []
    for s in d.segs:
        if not s.kind:
            continue
        at, dst, left = s.q * B + s.o, s.d, s.n
        while left:
            k = min(left, RB)
            segs.append(U.Segment(1, s.u, at // RB, at % RB, k, dst))
            at, dst, left = at + k, dst + k, left - k
        if dst > RB:
            raise ValueError(f"a frame's stream to byte {dst} doesn't fit rows of {RB}")
    return U.Descriptor(r=d.r, segs=tuple(segs))


def prepare(setup: U.Setup, RB: int | None = None, ports: Sequence[str] | None = None) -> Prepared:
    """v5's units and layout for a setup, built once for each row size: each unit's outbox in rows of RB (B unless
    given), and the descriptors in the format whose depths are those trees' (on rows of RB < B, `resplit`'s, in the
    narrowest format that holds them), registered under the layout's binding and seed. On rows of RB < B, `ports`
    (every port unless given) are the ports laid out, the others' positions `NONE`: a direction's own layout, on its
    own row size, where another direction's frames don't fit RB."""
    L = setup.layout
    B = L.fmt.B
    RB = B if RB is None else RB
    mine = tuple(L.ports if ports is None or RB == B else (p for p in L.ports if p in ports))
    got = setup.__dict__.get(("_v5", RB, mine))
    if got is None:
        if not 1 <= RB <= B:
            raise ValueError(f"rows are 1 ... B = {B} bytes, not {RB}")
        old = setup.units
        trees = {u: register_outbox(old.program, u, x, RB, old.seed) for u, x in old.data.items()}
        units = U.UnitTable(list(trees), tuple(t.depth for t in trees.values()), [t.root for t in trees.values()],
                            [t.domain_id for t in trees.values()], trees, dict(old.data), old.program, old.seed)
        if RB == B:
            fmt, descs = L.fmt._replace(DEPTHS=units.depths), L.descriptors
        else:
            descs = {p: [resplit(d, B, RB) for d in ds] for p, ds in L.descriptors.items() if p in mine}
            fmt = U.fit([d for ds in descs.values() for d in ds], RB, units.depths, L.fmt.DL)
        rows = [U.encode(fmt, descs[p][j] if j < len(descs.get(p, [])) else U.NONE) for p in L.ports
                for j in range(L.cap)]
        tree = U.register(rows, fmt.LW, L.tree.reg.binding, setup.seed + b"/layout")
        got = setup.__dict__[("_v5", RB, mine)] = Prepared(units, U.Layout(fmt, L.ports, L.cap, descs, tree))
    return got


# --- templates --------------------------------------------------------------------------------------------------------

def template(tpl, FB: int) -> tuple[int, int, int, bytes, bytes] | None:
    """A template static as (mode, p, s, prefix, suffix), or None for (); refuses one that isn't a placement in a
    frame of FB bytes."""
    if tpl == ():
        return None
    try:
        mode, p, s, pre, suf = tpl
        pre, suf = bytes.fromhex(pre), bytes.fromhex(suf)
    except (TypeError, ValueError):
        raise ValueError(f"TPL is () or (mode, p, s, prefix hex, suffix hex), not {tpl!r}") from None
    if (mode not in (U.RAW, U.B64) or not all(isinstance(x, int) and not isinstance(x, bool) for x in (p, s))
            or not 0 <= p <= FB or not 0 <= s <= FB or len(pre) > FB or s + len(suf) > FB):
        raise ValueError(f"a template is RAW or BASE64 with 0 <= p, s <= B and its prefix and suffix in the frame, "
                         f"not {tpl!r}")
    return mode, p, s, pre, suf


def tpl_of(mode: int, p: int, s: int, prefix: bytes, suffix: bytes) -> tuple:
    return (mode, p, s, prefix.hex(), suffix.hex())


def b64_chars(FB: int, p: int, L: int) -> int:
    """The base64 characters a template at p writes in a frame of FB bytes from a stream of L bytes: whole groups up
    to the frame's, cut at its end, and no more than L bytes encode to."""
    return min(FB - p, 4 * (FB // 4), 4 * -(-L // 3))


def place_fixed(tpl, FB: int, r: int, stream: bytes) -> bytes:
    """The frame of FB bytes the template puts on a stream of L bytes and r: the prefix, the stream's first min(r, L)
    bytes (RAW) or the base64 of its first r, zero past L, (BASE64, `b64_chars` of it) at p, and the suffix at s.
    With L = FB, `n3_unit.assemble`'s placement."""
    mode, p, s, pre, suf = template(tpl, FB)  # type: ignore[misc]
    L = len(stream)
    data = (int.from_bytes(stream, "little") & ((1 << 8 * min(r, L)) - 1)).to_bytes(L, "little")
    if mode == U.RAW:
        text = data[:FB - p]
    else:
        chars = b64_chars(FB, p, L)
        enc = base64.b64encode(data.ljust(3 * -(-chars // 4), b"\0")[:3 * -(-chars // 4)])
        out = bytearray(chars)
        for j in range(chars):
            if r >= 3 * j // 4 + 1:
                out[j] = enc[j]
            elif r >= 3 * (j // 4) + 1:
                out[j] = ord("=")
        text = bytes(out)
    x = int.from_bytes(pre, "little") ^ (int.from_bytes(text, "little") << 8 * p) ^ (int.from_bytes(suf, "little")
                                                                                     << 8 * s)
    return (x & ((1 << 8 * FB) - 1)).to_bytes(FB, "little")


def assemble_fixed(f: U.Format, tpl, row: bytes, rows: Sequence[bytes], G: int, FB: int | None = None) -> bytes:
    """The frame of FB bytes (B unless given) the template puts on the descriptor's r and segments' rows:
    `N3FrameFixed`'s reference, less its check that the segments from G on are of kind 0."""
    x = U.fields(f, row)
    stream = 0
    for g in range(G):
        src = rows[g] if g < len(rows) else bytes(2 * f.B)
        stream ^= int.from_bytes(U.segment(f.B, x.segs[g], src), "little")
    return place_fixed(tpl, f.B if FB is None else FB, x.r, stream.to_bytes(f.B, "little"))


# --- parts ------------------------------------------------------------------------------------------------------------

N3Next = U._part("N3Next", ("D",), lambda S: ((("q", Array(S.D, BIT)),), Array(S.D, BIT)),
                 lambda C, S, q: add(C, list(q), const_bits(1, S.D), width=S.D), "q + 1 mod 2^D")


def _build_runs_one(C, S, kind, mem):
    k = kind[0]
    seen, twice = mem[0], None
    for m in mem[1:]:
        both = C.AND(seen, m)
        twice = both if twice is None else twice ^ C.AND(twice.inv(), both)
        seen = seen ^ C.AND(seen.inv(), m)
    one = seen if twice is None else C.AND(seen, twice.inv())
    return [C.AND(k, m) for m in mem] + [C.AND(k, one.inv()).inv()]


N3RunsOne = U._part(
    "N3RunsOne", ("R",), lambda S: ((("kind", BIT), ("mem", Array(S.R, BIT))), Array(S.R + 1, BIT)), _build_runs_one,
    "a read's membership of each run (kind 1 and named), then its ok bit: kind 0, or in exactly one run")


def _build_advance(C, S, a, b, x, sel):
    s = sel[0]
    return [u ^ C.AND(s, u ^ v) for u, v in zip(a, b)] + [u ^ C.AND(s, u ^ v) for u, v in zip(b, x)]


N3Advance = U._part(
    "N3Advance", ("W",),
    lambda S: ((("a", Array(S.W, BIT)), ("b", Array(S.W, BIT)), ("x", Array(S.W, BIT)), ("sel", BIT)),
               Array(2 * S.W, BIT)),
    _build_advance, "a run's two rows after a read: (b, x) if it is a fresh member, else (a, b)")


def _frame_bytes(S, f: U.Format) -> int:
    FB = S.FB
    if not isinstance(FB, int) or isinstance(FB, bool) or FB < f.B:
        raise ValueError(f"FB is a frame's bytes, at least the rows' B = {f.B}, not {FB!r}")
    return FB


def _sig_place_fixed(S):
    f = U._format(S.FMT)
    FB = _frame_bytes(S, f)
    template(S.TPL, FB)
    if S.TPL == ():
        raise ValueError("N3PlaceFixed takes a template")
    return (("r", Array(f.LB, BIT)), ("stream", Array(8 * f.B, BIT)), ("frame", Array(FB, BYTE))), BIT


def _build_place_fixed(C, S, r, stream, *frame):
    f = U._format(S.FMT)
    FB = S.FB
    mode, p, s, pre, suf = template(S.TPL, FB)
    nb, room = 8 * FB, FB - p
    ge = U._thresholds(C, r)
    if mode == U.RAW:
        text = [C.AND(stream[k], ge(k // 8 + 1)) for k in range(8 * min(room, f.B))]
    else:
        chars = b64_chars(FB, p, f.B)
        g = -(-chars // 4)
        data = [C.AND(stream[k], ge(k // 8 + 1)) for k in range(min(8 * 3 * g, 8 * f.B))]
        enc = U.b64(C, data + [ZERO] * (8 * 3 * g - len(data)), 3 * g)
        text = []
        for j in range(chars):
            real = ge(3 * j // 4 + 1)
            pad = ge(3 * (j // 4) + 1) ^ real
            text += [C.AND(real, enc[8 * j + t]) ^ (pad if ord("=") >> t & 1 else ZERO) for t in range(8)]
    fixed = int.from_bytes(pre, "little") ^ (int.from_bytes(suf, "little") << 8 * s)
    want = const_bits(fixed, nb)
    for k, x in enumerate(text):
        want[8 * p + k] = want[8 * p + k] ^ x
    got = [x for byte in frame for x in byte]
    return [U._eq(C, want, got)]


N3PlaceFixed = U._part(
    "N3PlaceFixed", ("FMT", "TPL", "FB"), _sig_place_fixed, _build_place_fixed,
    "1 iff the FB-byte frame is the template's prefix XOR text << 8p XOR suffix << 8s, the text the B-byte stream's "
    "first min(r, B) bytes (RAW) or the base64 of its first r, zero past B (BASE64)")


def _sig_frame_fixed(S):
    f = U._format(S.FMT)
    if not isinstance(S.G, int) or not 1 <= S.G <= f.G:
        raise ValueError(f"G is 1 ... the format's {f.G} segments, not {S.G!r}")
    return (("descriptor", Array(f.LW, BIT)), ("rows", Array(S.G, Array(16 * f.B, BIT))),
            ("frame", Array(_frame_bytes(S, f), BYTE))), BIT


def _body_frame_fixed(B, S, desc, rows, frame):
    f = U._format(S.FMT)
    nb, d = 8 * f.B, desc.refs

    def each(off: int, w: int) -> Coll:
        return Coll(Array(S.G, Array(w, BIT)),
                    concat([d.slice(f.HW + g * f.SW + off, f.HW + g * f.SW + off + w) for g in range(S.G)]))

    segs = B.batch(bind(U.N3Seg, B=f.B), rows, each(f.IDW, f.OB), each(f.IDW + f.OB, f.LB),
                   each(f.IDW + f.OB + f.LB, f.LB), axes=(0, 0, 0, 0))
    stream = B.call(bind(U.N3Xor, N=S.G, W=nb), segs) if S.G > 1 else segs[0]
    ok = B.call(bind(N3PlaceFixed, FMT=S.FMT, TPL=S.TPL, FB=S.FB),
                Coll(Array(f.LB, BIT), d.slice(f.R_OFF, f.R_OFF + f.LB)), stream, frame)
    if S.G == f.G:
        return ok
    kinds = U._coll(Array(f.G - S.G, BIT), [d[f.HW + g * f.SW] for g in range(S.G, f.G)])
    return B.call(BL.and_gate(2), U._cat(Array(2, BIT), ok, B.call(bind(U.N3IsZero, W=f.G - S.G), kinds)))


N3FrameFixed = CompositeDefinition(
    "N3FrameFixed", 1, ("FMT", "G", "TPL", "FB"), _sig_frame_fixed, _body_frame_fixed,
    conformance="exact-model-tested", register=False,
    doc="1 iff the FB-byte frame is what the template puts on the descriptor's r and its first G segments' rows of B "
        "bytes: the segments' streams (N3Seg) XORed, placed by N3PlaceFixed; and every segment from G on is of kind 0")


# --- the window -------------------------------------------------------------------------------------------------------

def _sizes(S) -> tuple:
    f = U._format(S.FMT)
    n = S.N - S.SALTED
    T = n * S.G
    return f, n, T, min(T + 1, 1 << f.DB), V4._zw(T + 1)


def compaction(T: int, S: int) -> list[tuple[list[int], int]]:
    """Each stage b of a run's compaction network over its T reads into S slots, as (positions, W): the positions a
    slot below S still reads, from the last stage back (stage b's position p reads stage b - 1's p and p + 2^b), and
    the elements' width, as `n3_v4.compaction`'s. With S below T, a stage skips the positions between the ones read."""
    n = (T - 1).bit_length()
    need, out = set(range(min(S, T))), []
    for b in reversed(range(n)):
        out.append((sorted(need), 1 + n - b + 512))
        need |= {p + (1 << b) for p in need if p + (1 << b) < T}
    return out[::-1]


def _window_params(S) -> list[tuple[str, object]]:
    f, n, T, Sr, _ = _sizes(S)
    FB = _frame_bytes(S, f)
    if FB != f.B and S.TPL == ():
        raise ValueError(f"frames of FB = {FB} bytes on rows of B = {f.B} take a template")
    base = U._window_params(SimpleNamespace(FMT=tuple(f._replace(B=FB)), REC=S.REC, SALTED=S.SALTED, N=S.N, G=S.G))
    if not isinstance(S.R, int) or isinstance(S.R, bool) or S.R < 1:
        raise ValueError(f"N3Window_v5 takes R >= 1 runs, not {S.R!r}")
    if n > 1 << f.DL:
        raise ValueError(f"{n} descriptors don't fit a layout of 2^{f.DL} positions")
    template(S.TPL, FB)
    raw = Array(8 * f.B, BIT)
    params = []
    for k, t in base:
        if k == "descriptors":
            params.append((k, Array(n, Array(f.LW, BIT))))
        elif k == "descriptor_paths":
            params += [("descriptor_left", Array(f.DL, DIGEST)), ("descriptor_right", Array(f.DL, DIGEST))]
        elif k == "rows":
            params.append(("rows", Array(n, Array(S.G, raw))))
        elif k == "row_paths":
            params += [("runs", Array(T, Array(S.R, BIT))), ("fresh", Array(T, BIT)),
                       ("run_units", Array(S.R, Array(f.UB, BIT))), ("run_starts", Array(S.R, Array(f.DB, BIT))),
                       ("run_heads", Array(S.R, raw)), ("run_head_salts", Array(S.R, SALT_T)),
                       ("run_fill", Array(S.R, Array(Sr - 1, DIGEST))), ("run_left", Array(S.R, Array(f.DB, DIGEST))),
                       ("run_right", Array(S.R, Array(f.DB, DIGEST)))]
        else:
            params.append((k, t))
    return params


def _sig_window(S):
    return tuple(_window_params(S)), BIT


def _body_window(B, S, *args):
    f, n, T, Sr, ZW = _sizes(S)
    N, G, W, FB = S.N, S.G, 8 * f.B, S.FB
    a = dict(zip([k for k, _ in _window_params(S)], args))
    K = U._Consts(B)
    oks: list[Coll] = []
    frames = a["frames"]
    data = frames
    if S.SALTED:
        flat = frames.reshape(Array(N, Array(8 * FB, BIT)))
        slot = B.call(bind(U.N3Slot, N=N), a["slot"])
        picked = B.batch(bind(U.N3Gate, W=8 * FB), flat, slot[N:2 * N], axes=(0, 0))
        chosen = B.call(bind(U.N3Xor, N=N, W=8 * FB), picked) if N > 1 else picked[0]
        oks.append(B.call(bind(U.N3SaltEq, B=FB), chosen, a["salt"]))
        if n:
            routed = B.batch(bind(U.N3Mux, W=8 * FB), flat[0:n], flat[1:N], slot[0:n], axes=(0, 0, 0))
            data = routed.reshape(Array(n, Array(FB, BYTE)))
    if n:
        d = a["descriptors"]
        pos = B.call(bind(U.N3Positions, DL=f.DL, N=n), a["at"])
        leaves = B.batch(bind(V4.N3HidingLeaf, N_BITS=f.LW, D=f.DL), d, a["descriptor_salts"], a["layout_domain"], pos,
                         axes=(0, 0, None, 0), name="descriptors")
        lev = B.call(bind(V4.N3Range, S=n, D=f.DL), a["at"], leaves, a["layout_domain"], a["descriptor_left"],
                     a["descriptor_right"], name="descriptors")
        oks.append(B.call(bind(LmsEqual, N=64), lev[f.DL], a["layout"], name="descriptors"))
        oks.append(B.call(bind(V4.N3Fits, D=f.DL, N=n), a["at"], name="descriptors"))
        idw = [d.refs.slice(i * f.LW + f.HW + g * f.SW, i * f.LW + f.HW + g * f.SW + f.IDW)
               for i in range(n) for g in range(G)]
        kinds = Coll(Array(T, BIT), concat([r.slice(0, 1) for r in idw]))
        us = Coll(Array(T, Array(f.UB, BIT)), concat([r.slice(1, 1 + f.UB) for r in idw]))
        qs = Coll(Array(T, Array(f.DB, BIT)), concat([r.slice(1 + f.UB, f.IDW) for r in idw]))
        rows = a["rows"].reshape(Array(T, Array(W, BIT)))
        salts = a["row_salts"].reshape(Array(T, SALT_T))
        doms = B.batch(bind(U.N3Pick, U=f.U), us, a["domains"], axes=(0, None), name="rows")
        nxt = B.batch(bind(N3Next, D=f.DB), qs, axes=(0,), name="rows")
        rleaves = B.batch(bind(V4.N3HidingLeaf, N_BITS=W, D=f.DB), rows, salts, doms, nxt, axes=(0, 0, 0, 0),
                          name="rows")
        mem = B.batch(bind(N3RunsOne, R=S.R), kinds, a["runs"], axes=(0, 0), name="rows")
        oks.append(Coll(Array(T, BIT), concat([mem.refs.slice(t * (S.R + 1) + S.R, (t + 1) * (S.R + 1))
                                                for t in range(T)])))
        fresh = a["fresh"]
        zero_digest = U._coll(DIGEST, K.bits(0, 512))
        zero_row = U._coll(Array(W, BIT), K.bits(0, W))
        stages = compaction(T, Sr - 1)
        nz = len(stages)
        windows: list[list[Coll]] = [[] for _ in range(T)]
        for r in range(S.R):
            u_r, start = a["run_units"][r], a["run_starts"][r]
            domain = B.call(bind(U.N3Pick, U=f.U), u_r, a["domains"], name="rows")
            head = B.call(bind(V4.N3HidingLeaf, N_BITS=W, D=f.DB), a["run_heads"][r], a["run_head_salts"][r], domain,
                          start, name="rows")
            first, second = zero_row, a["run_heads"][r].reshape(Array(W, BIT))
            last, z, elems, carried = zero_digest, U._coll(Array(ZW, BIT), K.bits(0, ZW)), [], []
            for t in range(T):
                m = U._coll(BIT, [mem.refs[t * (S.R + 1) + r]])
                c = B.call(V4.N3Carry, last, rleaves[t], m, fresh[t], name="rows")
                carried.append(c.refs.slice(0, 1))
                sel = Coll(BIT, c.refs.slice(1, 2))
                elems.append(Coll(Array(1 + nz + 512, BIT),
                                  concat([sel.refs] + ([z.refs.slice(0, nz)] if nz else []) + [rleaves[t].refs])))
                pair = B.call(bind(N3Advance, W=W), first, second, rows[t], sel, name="rows")
                first, second = Coll(Array(W, BIT), pair.refs.slice(0, W)), Coll(Array(W, BIT), pair.refs.slice(W, 2 * W))
                windows[t].append(B.call(bind(U.N3Gate, W=2 * W), pair, m, name="rows"))
                if t < T - 1:
                    last = B.call(bind(U.N3Mux, W=512), last.reshape(Array(512, BIT)),
                                  rleaves[t].reshape(Array(512, BIT)), m, name="rows").reshape(DIGEST)
                z = B.call(bind(V4.N3Skip, W=ZW), z, sel, name="rows")
            oks.append(Coll(Array(T, BIT), concat(carried)))
            at = dict(enumerate(elems))
            for b, (ps, w) in enumerate(stages):
                s, invalid = 1 << b, U._coll(Array(w, BIT), K.bits(0, w))
                stay = Coll(Array(len(ps), Array(w, BIT)), concat([at[p].refs for p in ps]))
                inc = Coll(Array(len(ps), Array(w, BIT)),
                           concat([(at[p + s] if p + s < T else invalid).refs for p in ps]))
                out = B.batch(bind(V4.N3CompactStep, W=w), stay, inc, axes=(0, 0), name="rows")
                at = {p: out[i] for i, p in enumerate(ps)}
            elems = [at[j] for j in range(Sr - 1)]
            valid = Coll(Array(Sr - 1, BIT), concat([elems[j].refs.slice(0, 1) for j in range(Sr - 1)]))
            packed = Coll(Array(Sr - 1, Array(512, BIT)), concat([elems[j].refs.slice(1, 513) for j in range(Sr - 1)]))
            fill = a["run_fill"][r].reshape(Array(Sr - 1, Array(512, BIT)))
            placed = B.batch(bind(U.N3Mux, W=512), fill, packed, valid, axes=(0, 0, 0), name="rows")
            slots_ = Coll(Array(Sr, DIGEST), concat([head.refs, placed.refs]))
            lev = B.call(bind(V4.N3Range, S=Sr, D=f.DB), start, slots_, domain, a["run_left"][r], a["run_right"][r],
                         name="rows")
            oks.append(B.call(bind(V4.N3RunRoot, DEPTHS=f.DEPTHS), u_r, lev, a["roots"], name="rows"))
            oks.append(B.call(bind(V4.N3RunFits, DEPTHS=f.DEPTHS, T=T + 1, S=Sr), u_r, start, z, name="rows"))
        if S.R == 1:
            wins = [w[0] for w in windows]
        else:
            wins = [B.call(bind(U.N3Xor, N=S.R, W=2 * W), Coll(Array(S.R, Array(2 * W, BIT)),
                                                             concat([x.refs for x in w])), name="rows")
                    for w in windows]
        win = Coll(Array(n, Array(G, Array(2 * W, BIT))), concat([x.refs for x in wins]))
        if S.TPL == ():
            oks.append(B.batch(bind(U.N3Frame, FMT=S.FMT, G=G), d, win, data, axes=(0, 0, 0)))
        else:
            oks.append(B.batch(bind(N3FrameFixed, FMT=S.FMT, G=G, TPL=S.TPL, FB=FB), d, win, data, axes=(0, 0, 0)))
    rec = U._shape(S.REC)
    root = B.call(bind(U.N3FramesRoot, B=FB, N=N, D=rec.D), frames)
    hb = U._refs(a["head"])
    oks.append(B.call(bind(U.N3IsZero, W=8), U._coll(Array(8, BIT), hb[8 * rec.ST:8 * rec.ST + 8])))
    nh = 8 * (rec.HL + 64)
    head_ = U._coll(Array(nh, BIT), hb + [x for byte in U._digest_bytes(root) for x in byte])
    pos2 = U._coll(Array(2, BIT), U._refs(a["direction"]) + K.bits(1, 1))
    oks.append(B.call(bind(U.N3HidingRead, N_BITS=nh, D=2), head_, a["record_salt"], a["record"], a["record_domain"],
                      pos2, a["record_path"]))
    bits = concat([o.refs for o in oks])
    return B.call(BL.and_gate(bits.count), Coll(Array(bits.count, BIT), bits))


N3Window = CompositeDefinition(
    "N3Window", 5, ("FMT", "N", "G", "SALTED", "REC", "R", "TPL", "FB"), _sig_window, _body_window,
    conformance="exact-model-tested", register=False,
    doc="N3Window_v4's ok bit over outbox rows of B bytes: each read hashes the one row it adds to its run's two, its "
        "window those two; each run's head row opens before its fresh rows; with a template, every frame's placement "
        "is the template's and its descriptor gives only r and the segments, and frames may be FB > B bytes")


def unit(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, R: int, tpl: tuple = (), FB: int | None = None):
    return bind(N3Window, FMT=tuple(f), N=N, G=G, SALTED=int(salted), REC=tuple(rec), R=R, TPL=tuple(tpl),
                FB=f.B if FB is None else FB)


# --- instances and the reference --------------------------------------------------------------------------------------

def instance(setup: U.Setup, w: U.Window, R: int | None = None, tpl: tuple = (), RB: int | None = None,
             ports: Sequence[str] | None = None) -> U.Instance:
    """The window's v5 instance from its v3 one, against the setup's v5 units and layout on rows of RB (`prepare`, of
    `ports`; B unless given, and then a template's), with the port's G and runs on those rows."""
    v3 = w.instance
    if v3 is None:
        raise ValueError(f"{w.port}[{w.window}] runs no unit: {w.why}")
    prep = prepare(setup, RB, ports)
    narrow = prep.layout.fmt.B != setup.layout.fmt.B
    layout = prep.layout if narrow else setup.layout
    return convert(v3, prep, V4.runs_of(layout, w.port) if R is None else R, tpl,
                   G=layout.segments(w.port) if narrow else None)


def convert(v3: U.Instance, prep: Prepared, R: int, tpl: tuple = (), runs: Sequence[Sequence[int]] | None = None,
            fresh: Sequence[int] | None = None, starts: dict[int, int] | None = None,
            run_units: dict[int, int] | None = None, heads: dict[int, bytes] | None = None,
            rows: dict[int, bytes] | None = None, G: int | None = None) -> U.Instance:
    """The v5 instance of a v3 one, its witness edits kept: the descriptors re-encoded in v5's format; read t's row the
    second half of v3's row t (row q + 1) unless `rows` gives it, with its salt at q + 1; v4's runs and fresh bits
    unless given; each run's unit and start its first member's unless given, its head the first half of that member's
    v3 row (row start of an empty run) unless `heads` gives it; and the fill and siblings of the unit's tree at the
    start. On rows narrower than v3's frames (`prepare` with RB < B) the descriptors are `resplit`'s, G segments a
    frame (the most they need unless given), and each row and head the unit's own row in its tree, as v3 has none of
    that width."""
    f, n, FB = prep.layout.fmt, v3.N - v3.salted, v3.fmt.B
    B, narrow = f.B, f.B != v3.fmt.B
    descs = [U.decode(v3.fmt, x) for x in v3.args.get("descriptors", [])]
    if narrow:
        descs = [resplit(d, FB, B) for d in descs]
        G = max([1] + [len(d.segs) for d in descs]) if G is None else G
    else:
        G = v3.G
    units = prep.units
    args = {k: v for k, v in v3.args.items() if k not in ("descriptor_paths", "row_paths", "row_salts")}
    args.update(layout=prep.layout.tree.root, layout_domain=prep.layout.tree.domain_id, roots=list(units.roots),
                domains=list(units.domains))
    inst = U.Instance(f, v3.N, G, v3.salted, v3.rec, args)
    inst.R, inst.TPL, inst.FB = R, tuple(tpl), FB  # type: ignore[attr-defined]
    if not n:
        args.pop("rows", None)
        return inst
    at = args["at"]
    args["descriptors"] = [U.encode(f, d) for d in descs]
    args["descriptor_left"], args["descriptor_right"] = V4.siblings(prep.layout.tree.levels, at, n, f.DL)
    rs = V4.reads(f, args["descriptors"], G)
    T = len(rs)

    def tree_of(u: int) -> U.Tree | None:
        return units.trees[units.names[u]] if u < len(units.names) else None

    def own(u: int, q: int) -> bytes:
        t = tree_of(u)
        return t.reg.rows[q] if t is not None and q < len(t.reg.rows) else bytes(B)

    def salt(u: int, q: int) -> bytes:
        t = tree_of(u)
        return t.reg.salts[q] if t is not None and q < len(t.reg.salts) else NO_SALT

    if narrow:
        old = [own(s.u, s.q) + own(s.u, s.q + 1) if s.kind else bytes(2 * B) for s in rs]
    else:
        old = [r for per in v3.args["rows"] for r in per]
    flat = [(rows or {}).get(t, old[t][B:]) for t in range(T)]
    salts = [salt(s.u, s.q + 1) if s.kind else NO_SALT for s in rs]
    args["rows"] = [flat[i * G:(i + 1) * G] for i in range(n)]
    args["row_salts"] = [salts[i * G:(i + 1) * G] for i in range(n)]
    if runs is None:
        of, new = V4.partition(rs)
        runs, fresh = [[int(o == r) for r in range(R)] for o in of], [int(x) for x in new]
    Sr = min(T + 1, 1 << f.DB)
    args["runs"], args["fresh"] = [list(x) for x in runs], list(fresh)  # type: ignore[arg-type]
    units_, starts_, heads_, hsalts, fills, lefts, rights = [], [], [], [], [], [], []
    for r in range(R):
        members = [t for t in range(T) if rs[t].kind and runs[t][r]]
        u = (run_units or {}).get(r, rs[members[0]].u if members else 0)
        q0 = (starts or {}).get(r, rs[members[0]].q if members else 0)
        t_ = tree_of(u)
        levels = t_.levels if t_ is not None else ()
        leaves0 = levels[0] if levels else ()
        if r in (heads or {}):
            h = heads[r]  # type: ignore[index]
        elif members:
            h = old[members[0]][:B]
        else:
            h = t_.reg.rows[q0] if t_ is not None and q0 < len(t_.reg.rows) else bytes(B)
        count = sum(fresh[t] for t in members)  # type: ignore[index]
        fills.append([leaves0[q0 + 1 + j] if count <= j and q0 + 1 + j < len(leaves0) else ZD for j in range(Sr - 1)])
        left, right = V4.siblings(levels, q0, Sr, f.DB)
        units_.append(u)
        starts_.append(q0)
        heads_.append(h)
        hsalts.append(salt(u, q0))
        lefts.append(left)
        rights.append(right)
    args.update(run_units=units_, run_starts=starts_, run_heads=heads_, run_head_salts=hsalts, run_fill=fills,
                run_left=lefts, run_right=rights)
    return inst


def _assemble(f: U.Format, tpl: tuple, desc: bytes, rows: Sequence[bytes], G: int, FB: int) -> bytes:
    return U.assemble(f, desc, rows, G) if tpl == () else assemble_fixed(f, tpl, desc, rows, G, FB)


def reference(inst: U.Instance) -> bool:
    """`N3Window_v5`'s output on the instance, computed from its arguments as the circuit does; refuses a size class
    the unit refuses."""
    f, a, N, G, R = inst.fmt, inst.args, inst.N, inst.G, inst.R  # type: ignore[attr-defined]
    tpl, FB = inst.TPL, inst.FB  # type: ignore[attr-defined]
    unit_of(inst).signature()
    n, B = N - inst.salted, f.B
    frames = list(a["frames"])
    if inst.salted:
        slot = a["slot"]
        head = U.HEADER.pack(U.SALT, U.SALT_STREAM, U.SALT_PAYLOAD)
        if slot >= N or frames[slot] != (head + a["salt"]).ljust(FB, b"\0"):
            return False
        frames = frames[:slot] + frames[slot + 1:]
    descs = a.get("descriptors", [])
    windows: list[bytes] = []
    if n:
        at = a["at"]
        if at + n - 1 >= 1 << f.DL:
            return False
        leaves = [V4._leaf(a["layout_domain"], descs[i], f.LW, a["descriptor_salts"][i], at + i) for i in range(n)]
        if V4.range_levels(a["layout_domain"], at, leaves, a["descriptor_left"], a["descriptor_right"],
                           f.DL)[f.DL] != a["layout"]:
            return False
        rs = V4.reads(f, descs, G)
        rows = [r for per in a["rows"] for r in per]
        salts = [s for per in a["row_salts"] for s in per]
        T, Sr = len(rs), min(len(rs) + 1, 1 << f.DB)
        doms = [a["domains"][s.u] if s.u < f.U else ZD for s in rs]
        rl = [V4._leaf(doms[t], rows[t], 8 * B, salts[t], (rs[t].q + 1) % (1 << f.DB)) for t in range(T)]
        mem = [[int(s.kind and a["runs"][t][r]) for r in range(R)] for t, s in enumerate(rs)]
        if any(s.kind and sum(mem[t]) != 1 for t, s in enumerate(rs)):
            return False
        windows = [bytes(2 * B)] * T
        for r in range(R):
            u, start = a["run_units"][r], a["run_starts"][r]
            if u >= f.U:
                return False
            pair = (bytes(B), a["run_heads"][r])
            last, fresh_leaves = ZD, []
            for t in range(T):
                if not mem[t][r]:
                    continue
                if a["fresh"][t]:
                    fresh_leaves.append(rl[t])
                    pair = (pair[1], rows[t])
                elif rl[t] != last:
                    return False
                last = rl[t]
                windows[t] = pair[0] + pair[1]
            if len(fresh_leaves) + 1 > Sr or start + len(fresh_leaves) + 1 > 1 << f.DEPTHS[u]:
                return False
            head_leaf = V4._leaf(a["domains"][u], a["run_heads"][r], 8 * B, a["run_head_salts"][r], start)
            slots_ = [head_leaf] + fresh_leaves + list(a["run_fill"][r][len(fresh_leaves):])
            lev = V4.range_levels(a["domains"][u], start, slots_, a["run_left"][r], a["run_right"][r], f.DB)
            if lev[f.DEPTHS[u]] != a["roots"][u]:
                return False
    for i, x in enumerate(U.fields(f, d) for d in descs):
        mode_ok = x.mode in (U.RAW, U.B64) or tpl != ()
        if any(s.kind for s in x.segs[G:]) or not mode_ok or _assemble(f, tpl, descs[i], windows[i * G:(i + 1) * G],
                                                                       G, FB) != frames[i]:
            return False
    rec = inst.rec
    head = a["head"] + U.CT.frames_root(a["frames"], U.GridParams(rec.T, rec.R, FB))
    if head[rec.ST]:
        return False
    value = U.hiding_value(head, 8 * len(head), a["record_salt"])
    return U.root_ref(a["record_domain"], U.SCHEMA, value, U.N2.HEADS + a["direction"], a["record_path"]) == a["record"]


def arguments(inst: U.Instance) -> list[np.ndarray]:
    """The unit's arguments for one instance in its signature's order, each (1, leaves) of 0/1."""
    f, a = inst.fmt, inst.args
    digest = U._digest_bits
    flat: dict[str, list[int]] = {
        "layout": digest(a["layout"]), "layout_domain": digest(a["layout_domain"]), "at": U._int_bits(a["at"], f.DL),
        "record": digest(a["record"]), "record_domain": digest(a["record_domain"]), "direction": [a["direction"]],
        "roots": [b for r in a["roots"] for b in digest(r)], "domains": [b for r in a["domains"] for b in digest(r)],
        "frames": [b for fr in a["frames"] for b in U._bytes_bits(fr, 8 * inst.FB)],  # type: ignore[attr-defined]
        "head": U._bytes_bits(a["head"], 8 * inst.rec.HL), "record_salt": digest(a["record_salt"]),
        "record_path": [b for s in a["record_path"] for b in digest(s)],
    }
    if inst.salted:
        flat["salt"] = U._bytes_bits(a["salt"], 8 * U.SALT_PAYLOAD)
        flat["slot"] = U._int_bits(a["slot"], max(1, (inst.N - 1).bit_length()))
    if inst.N - inst.salted:
        flat["descriptors"] = [b for x in a["descriptors"] for b in U._bytes_bits(x, f.LW)]
        flat["descriptor_salts"] = [b for s in a["descriptor_salts"] for b in digest(s)]
        flat["descriptor_left"] = [b for s in a["descriptor_left"] for b in digest(s)]
        flat["descriptor_right"] = [b for s in a["descriptor_right"] for b in digest(s)]
        flat["rows"] = [b for per in a["rows"] for r in per for b in U._bytes_bits(r, 8 * f.B)]
        flat["row_salts"] = [b for per in a["row_salts"] for s in per for b in digest(s)]
        flat["runs"] = [b for per in a["runs"] for b in per]
        flat["fresh"] = list(a["fresh"])
        flat["run_units"] = [b for u in a["run_units"] for b in U._int_bits(u, f.UB)]
        flat["run_starts"] = [b for q in a["run_starts"] for b in U._int_bits(q, f.DB)]
        flat["run_heads"] = [b for r in a["run_heads"] for b in U._bytes_bits(r, 8 * f.B)]
        flat["run_head_salts"] = [b for s in a["run_head_salts"] for b in digest(s)]
        flat["run_fill"] = [b for per in a["run_fill"] for s in per for b in digest(s)]
        flat["run_left"] = [b for per in a["run_left"] for s in per for b in digest(s)]
        flat["run_right"] = [b for per in a["run_right"] for s in per for b in digest(s)]
    params, _ = unit_of(inst).signature()
    out = []
    for name, t in params:
        if len(flat[name]) != t.leaves:
            raise ValueError(f"argument {name} has {len(flat[name])} bits, the unit takes {t.leaves}")
        out.append(np.asarray(flat[name], dtype=np.uint8).reshape(1, -1))
    return out


def unit_of(inst: U.Instance):
    return unit(inst.fmt, inst.N, inst.G, inst.salted, inst.rec, inst.R,  # type: ignore[attr-defined]
                inst.TPL, inst.FB)  # type: ignore[attr-defined]


def evaluate(*insts: U.Instance) -> list[int]:
    """The unit's output on each instance, all of one unit, in one evaluation."""
    if len({(i.key, i.R, i.TPL, i.FB) for i in insts}) != 1:  # type: ignore[attr-defined]
        raise ValueError("the instances are of one unit")
    args = [np.concatenate(x) for x in zip(*(arguments(i) for i in insts))]
    return [int(b) for b in np.asarray(U.evaluate_bits(unit_of(insts[0]), *args)).reshape(-1)]


_FRAMES = ("N3Frame", "N3FrameFixed")


def size(target, memo: dict | None = None) -> dict:
    """The window unit's ANDs by part, as `n3_v4.size` groups v4's: the descriptors' run and everything the rows' runs
    compute, their windows included, are descriptor and row reads."""
    target = unit_of(target) if isinstance(target, U.Instance) else target
    memo = {} if memo is None else memo
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "segments", "frames", "frames_root",
                         "other"), 0)
    for nd in target.body.nodes:
        name = getattr(nd.fn, "name", "")
        group = V4._GROUPS.get(nd.name or "", U._GROUPS.get(name, "frames" if name in _FRAMES else "other"))
        if name == "N3HidingRead":
            group = "record_opening"
        ands = U._times(nd) * U._ands(nd.fn, memo)
        if name in _FRAMES:
            seg = U._times(nd) * sum(U._times(x) * U._ands(x.fn, memo) for x in nd.fn.body.nodes
                                     if getattr(x.fn, "name", "") == "N3Seg")
            out["segments"] += seg
            ands -= seg
        out[group] += ands
    out["ands"] = sum(out.values())
    return out


# --- circuit-check ----------------------------------------------------------------------------------------------------

def parts(inst: U.Instance) -> list[tuple[object, bool]]:
    """v5's new parts at the statics of the instance's window, each with whether circuit-check lowers it whole: the
    SHA-free ones whole, the B-byte row's leaf and the run's range by their parts. With a template, its placement and
    frame."""
    f, n, G, R = inst.fmt, inst.N - inst.salted, inst.G, inst.R  # type: ignore[attr-defined]
    tpl, FB = inst.TPL, inst.FB  # type: ignore[attr-defined]
    T = n * G
    Sr = min(T + 1, 1 << f.DB)
    whole = [bind(N3Next, D=f.DB), bind(N3RunsOne, R=R), bind(N3Advance, W=8 * f.B),
             bind(V4.N3RunFits, DEPTHS=f.DEPTHS, T=T + 1, S=Sr)]
    if any(V4.sel_js(Sr, f.DB, k) for k in range(f.DB)):
        whole.append(bind(V4.N3RangeSel, S=Sr, D=f.DB))
    if tpl != ():
        whole += [bind(N3PlaceFixed, FMT=tuple(f), TPL=tpl, FB=FB),
                  bind(N3FrameFixed, FMT=tuple(f), G=G, TPL=tpl, FB=FB)]
    hashed = [bind(V4.N3HidingLeaf, N_BITS=8 * f.B, D=f.DB), bind(V4.N3Range, S=Sr, D=f.DB)]
    return [(d, True) for d in whole] + [(d, False) for d in hashed]


def circuit_check(window: bool) -> dict:
    """circuit-check's checks on v5's new parts at the demo window's statics and at the canonical demo's (its
    templates' placement and frame, on rows of B and on rows of 11 under frames of 16) and, with `window`, on the
    demo's whole v5 window (lowered by its parts and not as a Call, as `n3_unit`'s command checks v3's)."""
    from circuit_check import checks as CK

    d = U.demo()
    inst = instance(d.setup, d.window)
    c = canonical_demo()
    templated = [instance(c.setup, w, tpl=c.templates[w.port], RB=RB) for RB in (None, 11) for w in c.windows]
    opts = CK.Options()
    entries = []
    targets = parts(inst) + [p for x in templated for p in parts(x) if p[0].name in ("N3PlaceFixed", "N3FrameFixed")]
    for target, whole in targets + ([(unit_of(inst), False)] if window else []):
        t0 = time.time()
        try:
            r = CK.check_definition(target, opts, lower_whole=whole, as_call=False)
        except Exception as e:  # noqa: BLE001  (a target the checks can't run on is a failure by name, as the CLI's)
            r = {"definition": target.id, "failures": [{"name": f"error/raised {target.id}: {type(e).__name__}: {e}"}],
                 "warnings": []}
        entries.append({**r, "kind": "definition", "target": target.id, "seconds": round(time.time() - t0, 3)})
        print(target.id, len(r.get("failures", [])), "failures", round(time.time() - t0, 1), "s", flush=True)
    rep = CK.report(entries, opts)
    rep["size"] = size(inst)
    rep["format"] = inst.fmt._asdict()
    return rep


# --- the canonical demo -----------------------------------------------------------------------------------------------

RAW_HEAD = U.HEADER.pack(U.DATA, 1, 7)
B64_HEAD = U.HEADER.pack(U.DATA, 2, 7)
NL2 = b"\n\n"


def _canon_frames(off_template: bool) -> tuple[list[list[tuple]], list[list[tuple]]]:
    """p0's frames, RAW: the template's header, up to 5 outbox bytes at 9 and "\\n\\n" at 14; p1's, BASE64: the
    template's header, the base64 of up to 3 bytes at 9 and "\\n\\n" at 13. With `off_template`, p0's second frame's
    header has another stream, which its descriptor holds and the template doesn't."""
    head = U.HEADER.pack(U.DATA, 5, 7) if off_template else RAW_HEAD
    p0 = [
        [U.const(RAW_HEAD), ("r", ("u0",), 8 * 14, 8 * 5), U.const(NL2)],
        [U.const(head), ("r", ("u0",), 8 * 33, 8 * 3), U.zeros(2), U.const(NL2)],
        [U.const(RAW_HEAD), ("r", ("u1",), 0, 8 * 2), ("r", ("u0",), 8 * 20, 8 * 3), U.const(NL2)],
    ]
    p1 = [
        [U.const(B64_HEAD), ("b", (("r", ("u1",), 0, 8), ("r", ("u0",), 8 * 20, 16)), 3), U.const(NL2), U.zeros(1)],
        [U.const(B64_HEAD), ("b", (("r", ("u0",), 8 * 15, 16),), 2), U.const(NL2), U.zeros(1)],
    ]
    return p0, p1


TEMPLATES = {"p0": tpl_of(U.RAW, 9, 14, RAW_HEAD, NL2), "p1": tpl_of(U.B64, 9, 13, B64_HEAD, NL2)}


@dataclass
class Canonical:
    setup: U.Setup
    windows: list[U.Window]
    templates: dict[str, tuple]


def canonical_demo(seed: bytes = b"n3-demo", off_template: bool = False,
                   flipped: tuple[str, int, int] | None = None) -> Canonical:
    """The demo's units on frames of a canonical framing: each port's frames one template's (`TEMPLATES`), every free
    field fixed, only r and the segments their own. p0's three RAW frames (u0's bytes 14 ... 18 across its rows 0
    and 1; three bytes of row 2; two units' bytes), p1's two BASE64 ones (two units'; u0's bytes 15 and 16 across rows
    0 and 1), each port's one window in two buckets of two slots. With `flipped` (port, frame, byte), that byte's low
    bit is flipped before the certifier records it."""
    B = 16
    outboxes = {u: hashlib.shake_256(seed + u.encode()).digest(k) for u, k in (("u0", 40), ("u1", 10))}
    units = U.UnitTable.register(outboxes, B, b"demo", seed)
    p0, p1 = _canon_frames(off_template)
    layout = U.register_layout({"p0": p0, "p1": p1}, ["p0", "p1"], B, units, b"demo", b"", seed)
    links = {"p0": ("demo", 0), "p1": ("demo", 1)}
    setup = U.Setup(None, units, layout, links, dict.fromkeys(links, U.GridParams(2, 2, B)), seed)  # type: ignore[arg-type]
    frames = {p: [U.expand(x, B, units.outbox) for x in fs] for p, fs in (("p0", p0), ("p1", p1))}
    if flipped is not None:
        port, i, at = flipped
        x = frames[port][i]
        frames[port][i] = x[:at] + bytes([x[at] ^ 1]) + x[at + 1:]
    windows = setup.attach([U.Window("p0", 0, frames["p0"], (2, 1)), U.Window("p1", 0, frames["p1"], (1, 1))])
    return Canonical(setup, windows, dict(TEMPLATES))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    c = sub_.add_parser("circuit-check", help="circuit-check's checks on v5's new parts and, with --window, its window")
    c.add_argument("--window", action="store_true", help="the demo's whole v5 window too")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rep = circuit_check(args.window)
    args.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
