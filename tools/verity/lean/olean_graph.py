"""Whether two builds of one module are one build but for where its source was: the module's `.olean` parts (`olean`,
and a `module` file's `olean.server` and `olean.private`) compared as the object graphs Lean's compactor wrote, not as bytes.

Lean 4.34's output is byte-identical for most modules' sources built twice at one path, whatever the thread count, but
not for every one's: an equation lemma Lean realizes when a declaration first asks for it can name its binders in that
declaration's hygiene context, so which declaration asks first changes the bytes (`Spot.SecondStage`'s `check.eq_3`),
which a records run meets by compiling the module again (`lean_changed.mismatched`). Nor is it at another path: a module's `olean.server` keeps each message its compile logged (a
linter's warning) with the absolute path of its source, and its `olean.private` points into that part, so a longer path
moves the objects after it. A
sampled recompile (`check.py`'s `sampled`) checks a module built in another tree, so for a module whose compile logged a
message, bytes alone would refuse it. `differs` walks both graphs from each part's root in step: every object must have
the same kind, fields and scalar bytes, and its fields the same, recursively, across the parts, which point into each
other. The one difference allowed is a string that, in both, is an absolute path ending in the module's own source path,
its length its own. Every declaration, extension entry and message is compared, so nothing but that path can differ.

`canonical` is a digest of one build's graph that is the same for two builds of a module at two paths, so the replay cache
(`check.py`'s `replay_keys`) keys a module by what it declares rather than by where it was compiled. It is narrower than
`differs`: each part's root is Lean's `ModuleData`, and what its imports and declarations (`MODULE_DATA` before `entries`)
reach is hashed as it is; only a string reached through the environment extensions' entries alone, where a message keeps
its file, may be the module's source path in either build, so the kernel replay (`Replay.lean`, which reads the imports
and the constants) never reads a string the digest left out.

The layout is Lean's (`src/include/lean/lean.h`, `src/runtime/compact.cpp`): a header of `HEADER` bytes, the base address
the part is mapped at, the root object's address, then the objects, each an 8-byte header (reference count, size, the
number of object fields, kind) and its body; an address with its low bit set is a boxed scalar. Stdlib only, loaded by
path."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

#: the bytes before the base address: the magic `olean`, the format's version, its flags, Lean's version and githash
HEADER = 80
BASE, ROOT, DATA = 80, 88, 96
#: a module's parts, each a compacted region of its own that the others may point into
PARTS = ("olean", "olean.server", "olean.private")
#: object kinds (`lean.h`): constructors are those below `MAX_CTOR`
MAX_CTOR, ARRAY, SCALAR_ARRAY, STRING, MPZ, THUNK, TASK, REF = 244, 246, 248, 249, 250, 251, 252, 253
#: the object fields of each part's root, Lean 4.34's `ModuleData` constructor (tag 0), in order
MODULE_DATA = ("imports", "constNames", "constants", "extraConstNames", "entries")
_HEAD = struct.Struct("<iHBB")
_U64 = struct.Struct("<Q")


class Malformed(ValueError):
    """A part that isn't a compacted region as Lean writes one, or an address outside every part."""


class _Region:
    """One build's parts, by the address range each is mapped at: no two ranges overlap, so an address is in one part."""

    def __init__(self, parts: dict[str, bytes]):
        self.parts = {}
        for name, data in parts.items():
            if len(data) < DATA or data[:5] != b"olean":
                raise Malformed(f"{name}: not an olean file")
            self.parts[name] = (_U64.unpack_from(data, BASE)[0], data)
        spans = sorted((base, base + len(data), name) for name, (base, data) in self.parts.items())
        for (_, end, a), (start, _, b) in zip(spans, spans[1:]):
            if start < end:
                raise Malformed(f"{a} and {b} are mapped over each other")

    def root(self, name: str) -> int:
        return _U64.unpack_from(self.parts[name][1], ROOT)[0]

    def at(self, addr: int) -> tuple[bytes, int]:
        for base, data in self.parts.values():
            if base + DATA <= addr < base + len(data):
                return data, addr - base
        raise Malformed(f"address {addr:#x} is in no part")


def _u64(data: bytes, off: int) -> int:
    return _U64.unpack_from(data, off)[0]


def _bytes(data: bytes, off: int, n: int) -> bytes:
    if off < 0 or n < 0 or off + n > len(data):
        raise Malformed(f"{n} bytes at {off} run past the part's {len(data)}")
    return data[off:off + n]


def _string(data: bytes, off: int) -> tuple[int, bytes]:
    """A string object's length in code points (`m_length`, which compiled `String.length` reads) and its bytes, its
    terminating NUL dropped."""
    return _u64(data, off + 24), _bytes(data, off + 32, _u64(data, off + 8))[:-1]


def _path_of_source(length: int, s: bytes, suffix: str) -> bool:
    try:
        text = s.decode()
    except UnicodeDecodeError:
        return False
    return (len(text) == length and text.startswith("/") and text.endswith(suffix) and "\0" not in text
            and "\n" not in text)


def differs(restored: dict[str, Path], rebuilt: dict[str, Path], source: str) -> str | None:
    """Why ``restored`` and ``rebuilt`` ({part: its file}, `PARTS`) are not one build of a module whose source path is
    ``source`` (as its name gives it: `Proofs/Foo.lean` for `Proofs.Foo`; Lake joins a package's `srcDir` without
    normalizing it, so nothing before it is compared), or None when they are: the same parts, with headers equal byte for
    byte, whose graphs are equal object for object from each part's root, but for strings that in both are absolute paths
    ending in ``/`` + ``source``, each of the length its text has."""
    if sorted(restored) != sorted(rebuilt):
        return f"parts {sorted(restored)} restored, {sorted(rebuilt)} rebuilt"
    suffix = "/" + source.lstrip("/")
    try:
        x = _Region({p: f.read_bytes() for p, f in restored.items()})
        y = _Region({p: f.read_bytes() for p, f in rebuilt.items()})
        for p in restored:
            if x.parts[p][1][:BASE] != y.parts[p][1][:BASE]:
                return f"{p}: the headers differ (format, flags or Lean's version)"
        seen: set[tuple[int, int]] = set()
        todo = [(x.root(p), y.root(p), p) for p in sorted(restored)]
        while todo:
            ax, ay, part = todo.pop()
            if (ax, ay) in seen:
                continue
            seen.add((ax, ay))
            if ax & 1 or ay & 1 or not ax or not ay:
                if ax != ay:
                    return f"{part}: a scalar or null field differs ({ax:#x}, {ay:#x})"
                continue
            (dx, ox), (dy, oy) = x.at(ax), y.at(ay)
            hx, hy = _HEAD.unpack_from(dx, ox), _HEAD.unpack_from(dy, oy)
            _, size, n, tag = hx
            if tag != hy[3]:
                return f"{part}: an object of kind {tag} restored where the rebuild has kind {hy[3]}"
            if tag < MAX_CTOR:
                if hx[1:] != hy[1:] or size < 8 + 8 * n:
                    return f"{part}: a constructor of tag {tag} with other fields ({hx[1:]} restored, {hy[1:]} rebuilt)"
                if _bytes(dx, ox + 8 + 8 * n, size - 8 - 8 * n) != _bytes(dy, oy + 8 + 8 * n, size - 8 - 8 * n):
                    return f"{part}: a constructor of tag {tag} with other scalar fields"
                todo.extend((_u64(dx, ox + 8 + 8 * i), _u64(dy, oy + 8 + 8 * i), part) for i in range(n))
            elif tag == ARRAY:
                k = _u64(dx, ox + 8)
                if k != _u64(dy, oy + 8):
                    return f"{part}: an array of {k} elements restored, {_u64(dy, oy + 8)} rebuilt"
                todo.extend((_u64(dx, ox + 24 + 8 * i), _u64(dy, oy + 24 + 8 * i), part) for i in range(k))
            elif tag == SCALAR_ARRAY:
                kx, ky = _u64(dx, ox + 8), _u64(dy, oy + 8)
                if n != hy[2] or kx != ky or _bytes(dx, ox + 24, n * kx) != _bytes(dy, oy + 24, n * ky):
                    return f"{part}: a scalar array differs"
            elif tag == STRING:
                sx, sy = _string(dx, ox), _string(dy, oy)
                if sx != sy and not (_path_of_source(*sx, suffix) and _path_of_source(*sy, suffix)):
                    return (f"{part}: a string differs ({sx[1][:80]!r} of length {sx[0]} restored, {sy[1][:80]!r} of "
                            f"length {sy[0]} rebuilt)")
            elif tag == MPZ:
                (_, kx), (_, ky) = struct.unpack_from("<ii", dx, ox + 8), struct.unpack_from("<ii", dy, oy + 8)
                if kx != ky:
                    return f"{part}: a big number differs"
                (lx, lox), (ly, loy) = x.at(_u64(dx, ox + 16)), y.at(_u64(dy, oy + 16))
                if _bytes(lx, lox, 8 * abs(kx)) != _bytes(ly, loy, 8 * abs(ky)):
                    return f"{part}: a big number differs"
            elif tag in (THUNK, TASK, REF):
                todo.append((_u64(dx, ox + 8), _u64(dy, oy + 8), part))
            else:
                return f"{part}: an object of kind {tag}, which a compacted region doesn't hold"
    except (Malformed, struct.error, OSError) as e:
        return f"malformed: {e}"
    return None


def canonical(files: dict[str, Path], source: str) -> str | None:
    """A digest of one build of a module ({part: its file}, `PARTS`) whose source path is ``source`` (as for `differs`)
    that two builds share when they are one but for where the source was: each part's name and header, then every object
    reachable from the parts' roots, each as the number its first reference gave it, its kind, its fields (an object's
    number, a boxed scalar or null) and its scalar bytes. The walk takes what the roots' imports and declarations
    (`MODULE_DATA` before `entries`) reach before anything only the entries reach, and a string of the second kind that is
    an absolute path ending in ``/`` + ``source``, of the length its text has, counts as such a path, whatever it is.
    None when a part isn't a compacted region whose root is a `ModuleData`."""
    suffix = "/" + source.lstrip("/")
    try:
        names = [p for p in PARTS if p in files]
        r = _Region({p: files[p].read_bytes() for p in names})
        spans = [(base + DATA, base + len(data), data) for base, data in r.parts.values()]

        def at(a: int) -> tuple[bytes, int]:
            for lo, hi, data in spans:
                if lo <= a < hi:
                    return data, a - lo + DATA
            raise Malformed(f"address {a:#x} is in no part")

        roots = {r.root(p) for p in names}
        number: dict[int, int] = {}
        exact: set[int] = set()
        now: list[int] = []
        later: list[int] = []

        def ref(a: int, entries: bool) -> bytes:
            if a & 1 or not a:
                return b"s" + _U64.pack(a)
            if (i := number.get(a)) is None:
                i = number[a] = len(number)
                if entries:
                    later.append(a)
            if not entries and a not in exact:
                exact.add(a)
                now.append(a)
            return b"o" + _U64.pack(i)

        def record(a: int, entries: bool) -> bytes:
            d, o = at(a)
            _, size, n, tag = _HEAD.unpack_from(d, o)
            if tag < MAX_CTOR:
                if size < 8 + 8 * n:
                    raise Malformed(f"a constructor of {size} bytes with {n} object fields")
                if a in roots and (tag != 0 or n != len(MODULE_DATA)):
                    raise Malformed(f"a root of tag {tag} with {n} object fields, not a `ModuleData`")
                fs = struct.unpack_from(f"<{n}Q", d, o + 8)
                last = len(MODULE_DATA) - 1 if a in roots else n
                rec = (struct.pack("<BBH", tag, n, size) + b"".join(ref(f, entries or i >= last) for i, f in enumerate(fs))
                       + _bytes(d, o + 8 + 8 * n, size - 8 - 8 * n))
            elif tag == ARRAY:
                k = _u64(d, o + 8)
                rec = struct.pack("<BQ", tag, k) + b"".join(ref(f, entries) for f in struct.unpack_from(f"<{k}Q", d, o + 24))
            elif tag == SCALAR_ARRAY:
                k = _u64(d, o + 8)
                rec = struct.pack("<BBQ", tag, n, k) + _bytes(d, o + 24, n * k)
            elif tag == STRING:
                length, s = _string(d, o)
                rec = bytes([tag]) + (b"path" if entries and _path_of_source(length, s, suffix)
                                      else struct.pack("<QQ", length, len(s)) + s)
            elif tag == MPZ:
                k = struct.unpack_from("<ii", d, o + 8)[1]
                rec = struct.pack("<Bi", tag, k) + _bytes(*at(_u64(d, o + 16)), 8 * abs(k))
            elif tag in (THUNK, TASK, REF):
                rec = bytes([tag]) + ref(_u64(d, o + 8), entries)
            else:
                raise Malformed(f"an object of kind {tag}, which a compacted region doesn't hold")
            return _U64.pack(number[a]) + rec

        h = hashlib.sha256()
        for p in names:
            h.update(p.encode() + b"\0" + r.parts[p][1][:BASE])
        h.update(b"".join(ref(r.root(p), False) for p in names))
        for a in now:
            rec = record(a, False)
            h.update(_U64.pack(len(rec)) + rec)
        for a in later:
            if a not in exact:
                rec = record(a, True)
                h.update(_U64.pack(len(rec)) + rec)
    except (Malformed, struct.error, OSError):
        return None
    return h.hexdigest()
