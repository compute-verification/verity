"""`class_statement.stage_unit` in less time and memory, writing the same bytes: `install(scratch)` patches the library for
one process (stage.py's); `composed()`, `size()` and `release()` hand stage.py the circuit it composed, so it never reads
the circuit back.

    - the cycle collector is off: staging makes no cyclic garbage (a K=256 row unit's 30k collections freed 325 objects),
      and its collections were about 30% of the time at K=256, more at K=4096
    - the unit circuit's ANDs keep their operand forms as tuples, not frozensets, and drop them once the unit is laid out:
      nothing reads them after `ir_lower.layout`
    - the unit's netlist (`ir_lower.netlist`'s text) is written once, to a file under ``scratch``, hashed as it is written:
      `compose` builds it as one string, which `Lowering.sha256_of` hashes with the LEAVES and CUT lines. `compose` gets
      in its place a string of as many newlines (and `sha256_of` the file's hash), and the circuit text it returns stands
      for that text with the file in the newlines' place: its digests (`circuit.digest`, `circuit.class_digest`) and
      circuit.txt are streamed from the parts, both digests in the pass that writes the file
    - any other text's SHA-512 / SHA-256 goes in pieces of 64 Mi characters, each computed once
    - the unit's layout (`ands_layout`) sorts each form's columns in one call, and `ir_lower.evaluate_layout` XORs a row's
      columns with `functools.reduce`: the same rows and values

stage.py runs the library unpatched under `HZ_STAGE=library`; both write the same bytes."""
from __future__ import annotations

import gc
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from operator import xor
from pathlib import Path

from verity.ml.circuits.boolean import forms
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import ir_lower as IL

CHUNK = 1 << 26
ROWS = 1 << 16
_scratch: list[Path] = []
_memo: dict[int, tuple[str, dict]] = {}
_unit_lays: dict[int, object] = {}
_units: dict[tuple[int, str], dict] = {}
_streams: dict[int, dict] = {}
_in_unit: list = []
_lowering: list = []
_composed: list = []


def _update(h, text: str, start: int = 0, stop: int | None = None):
    stop = len(text) if stop is None else stop
    for i in range(start, stop, CHUNK):
        h.update(text[i:min(i + CHUNK, stop)].encode())
    return h


def _of(text: str) -> dict:
    e = _memo.get(id(text))
    if e is None or e[0] is not text:
        e = _memo[id(text)] = (text, {})
    return e[1]


def _class_head(text: str) -> tuple[bytes, int]:
    """`circuit.class_digest`'s first two lines (META without its binding keys), and where the rest starts."""
    i = text.index("\n")
    j = text.index("\n", i + 1)
    meta = {k: v for k, v in json.loads(text[i + 6:j]).items() if k not in FC.BINDING_KEYS}
    return (text[:i] + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n").encode(), j + 1


def _stream(text: str, out=None) -> dict:
    """A composed text's file digest and class digest, in one pass over its parts, writing them to ``out`` too."""
    s = _streams[id(text)]
    head, j = _class_head(text)
    h_file, h_class = hashlib.sha512(), hashlib.sha512(head)
    with ThreadPoolExecutor(2) as pool:
        def put(b: bytes, file: bool = True, cls: bool = True):
            jobs = [pool.submit(h.update, b) for h, on in ((h_file, file), (h_class, cls)) if on]
            if out is not None:
                out.write(b)
            for f in jobs:
                f.result()

        put(text[:j].encode(), cls=False)
        put(text[j:s["at"]].encode())
        with open(s["path"], "rb") as f:
            while b := f.read(CHUNK):
                put(b)
        put(text[s["at"] + s["newlines"]:].encode())
    return {"sha512": h_file.hexdigest(), "class": h_class.hexdigest()}


def digest(text: str) -> str:
    """`circuit.digest`."""
    m = _of(text)
    if "sha512" not in m:
        if id(text) in _streams:
            m.update(_stream(text))
        else:
            m["sha512"] = _update(hashlib.sha512(), text).hexdigest()
    return m["sha512"]


def class_digest(text: str) -> str:
    """`circuit.class_digest`."""
    m = _of(text)
    if "class" not in m:
        if id(text) in _streams:
            m.update(_stream(text))
        else:
            head, j = _class_head(text)
            m["class"] = _update(hashlib.sha512(head), text, j).hexdigest()
    return m["class"]


def sha256(text: str) -> str:
    """`ir_lower.digest`."""
    return _update(hashlib.sha256(), text).hexdigest()


def _netlist(lay, name: str):
    """`ir_lower.netlist(lay, name)`'s text in pieces."""
    head = [IL.FORMAT, name, lay.useful, lay.const]
    for groups in (lay.in_groups, lay.out_groups):
        head.append(len(groups))
        for c, words, bits in groups:
            head += [c, words, len(bits), *bits]
    yield " ".join(map(str, head)) + "\n"
    cols, rows = IL._cols_of, []
    for a, b in zip(lay.rows_a, lay.rows_b):
        rows.append(" ".join(map(str, [len(a), *cols(a), len(b), *cols(b)])))
        if len(rows) == ROWS:
            yield "\n".join(rows) + "\n"
            rows = []
    if rows:
        yield "\n".join(rows) + "\n"


def _unit(lay, name: str) -> dict:
    """The unit's expanded netlist, written once to a scratch file: its path, newlines, bytes and sha256 state."""
    e = _units.get((id(lay), name))
    if e is None or e["lay"] is not lay:
        path = _scratch[0] / f".unit-{len(_units)}.netlist"
        h, n, size = hashlib.sha256(), 0, 0
        with open(path, "wb") as f:
            for piece in _netlist(lay, name):
                b = piece.encode()
                h.update(b)
                f.write(b)
                n += piece.count("\n")
                size += len(b)
        e = _units[(id(lay), name)] = {"lay": lay, "path": path, "newlines": n, "bytes": size, "sha256": h}
    return e


def _lowering_sha256(self) -> str:
    d = self.__dict__.get("_sha256")
    if d is None:
        tail = "LEAVES " + json.dumps(self.leaves, sort_keys=True, separators=(",", ":")) + "\n"
        cut = self.cut
        if cut is not None:
            tail += "CUT " + json.dumps(cut, sort_keys=True, separators=(",", ":")) + "\n"
        if id(self.layout) in _unit_lays:
            h = _unit(self.layout, self.name)["sha256"].copy()
        else:
            h = _update(hashlib.sha256(), FC._expanded(self.layout, self.name))
        h.update(tail.encode())
        d = self.__dict__["_sha256"] = h.hexdigest()
    return d


def _lowering_sha256_of(self, expanded: str) -> str:
    """`ir_lower.Lowering.sha256_of`, whose ``expanded`` is the newlines `compose` got in the unit netlist's place."""
    return self.sha256


def ands_layout(C, in_names, out_names, in_split=None, out_split=None):
    """`ir_lower.layout` of a circuit of inputs and ANDs only: the same rows, each form's columns sorted in one call (every
    form's bits are distinct and laid out, so the library's parity pass keeps them all) and the constant column, the last,
    appended."""
    W = IL.WORD
    live = C.live()
    col = [-1] * len(C.kind)
    rows_a: list = []
    rows_b: list = []

    def aligned(n: int, nbits: int) -> tuple[int, int]:
        words = IL.ceil_pow2(max(1, -(-nbits // W)))
        return -(-n // (W * words)) * (W * words), words

    def pad(to: int) -> None:
        rows_a.extend([] for _ in range(to - len(rows_a)))
        rows_b.extend([] for _ in range(to - len(rows_b)))

    in_groups = []
    for g in IL._groups(in_names, in_split):
        bits = [len(C.inputs[nm]) for nm in g]
        at, words = aligned(len(rows_a), sum(bits))
        pad(at)
        for nm in g:
            for w in C.inputs[nm]:
                (v,) = tuple(w.s)
                col[v] = i = len(rows_a)
                rows_a.append([i])
                rows_b.append([i])
        pad(at + words * W)
        in_groups.append((at // W, words, bits))
    ands = [v for v, kind in enumerate(C.kind) if kind == "and" and v in live]
    first = len(rows_a)
    for i, v in enumerate(ands, first):
        col[v] = i
    n = first + len(ands) + len(C.asserts)
    out_at = []
    for g in IL._groups(out_names, out_split):
        bits = [len(C.outputs[nm]) for nm in g]
        at, words = aligned(n, sum(bits))
        out_at.append((g, at, words, bits))
        n = at + words * W
    const = n
    get = col.__getitem__

    def form(Lf) -> list:
        r = sorted(map(get, Lf.s))
        if Lf.c & 1:
            r.append(const)
        return r

    data = C.data
    for v in ands:
        a, b = data[v]
        rows_a.append(form(a))
        rows_b.append(form(b))
    for _, Lf in C.asserts:
        rows_a.append(sorted(form(Lf) + [len(rows_a)]))
        rows_b.append([const])
    out_groups = []
    for g, at, words, bits in out_at:
        pad(at)
        for nm in g:
            for w in C.outputs[nm]:
                rows_a.append(form(w))
                rows_b.append([const])
        pad(at + words * W)
        out_groups.append((at // W, words, bits))
    assert len(rows_a) == const
    rows_a.append([const])
    rows_b.append([const])
    lay = IL.LoweredUnit(rows_a, rows_b, const, in_groups, out_groups, {})
    lay.counts = {"ands": len(ands), "asserts": len(C.asserts), "rows": len(rows_a), "in_words": lay.in_words,
                  "out_words": lay.out_words, "nnz_a": sum(map(len, rows_a)), "nnz_b": sum(map(len, rows_b)), "reads": 0,
                  "read_rows": 0, "read_nnz_a": 0}
    return lay


def evaluate_layout(lay, ports, T, flip=None):
    """`ir_lower.evaluate_layout`."""
    mask = (1 << T) - 1
    z = [0] * lay.useful
    for at, width, words in zip(lay.in_port_cols, lay.in_bits, ports):
        z[at:at + width] = IL.slice_words(words, width)
    z[lay.const] = mask
    get, cols, ra, rb = z.__getitem__, IL._cols_of, lay.rows_a, lay.rows_b
    for i in range(lay.in_words * IL.WORD, lay.useful):
        if i != lay.const:
            z[i] = reduce(xor, map(get, cols(ra[i])), 0) & reduce(xor, map(get, cols(rb[i])), 0)
    if flip is not None:
        z[flip[0]] ^= 1 << flip[1]
    ok = mask
    for i in range(lay.useful):
        ok &= ~((reduce(xor, map(get, cols(ra[i])), 0) & reduce(xor, map(get, cols(rb[i])), 0)) ^ z[i]) & mask
    outs = [IL.unslice_words(z[at:at + nb], T) for at, nb in zip(lay.out_port_cols, lay.out_bits)]
    return outs, ok


def install(scratch: Path) -> None:
    """Patch the library for this process; the unit's netlist goes under ``scratch`` until `release`. The slot circuits'
    caches (`cache_slot_circuits`) go first, so these wrap them."""
    gc.disable()
    scratch.mkdir(parents=True, exist_ok=True)
    _scratch[:] = [scratch]
    CS.cache_slot_circuits()
    new, lower, layout, expanded = forms.Circuit._new, IL.lower, IL.layout, FC._expanded
    unit_lowering, compose, write_text = CS.unit_lowering, FC.compose, Path.write_text

    def new_(self, kind, data):
        if kind == "and" and _lowering and self is _lowering[-1]:
            a, b = data
            data = (forms.L(tuple(a.s), a.c), forms.L(tuple(b.s), b.c))
        return new(self, kind, data)

    def unit_lowering_(*a, **k):
        _in_unit.append(True)
        try:
            low, out_words = unit_lowering(*a, **k)
        finally:
            _in_unit.pop()
        if not hasattr(low, "template"):
            _unit_lays[id(low.layout)] = low.layout
        return low, out_words

    def lower_(C, *a, **k):
        if not _in_unit:
            return lower(C, *a, **k)
        C._compact = True
        _lowering.append(C)
        try:
            return lower(C, *a, **k)
        finally:
            _lowering.pop()

    def layout_(C, *a, **k):
        compact = getattr(C, "_compact", False)
        lay = (ands_layout if compact and set(C.kind) <= {"in", "and"} else layout)(C, *a, **k)
        if compact:
            C.data = [None if kind == "and" else d for kind, d in zip(C.kind, C.data)]
        return lay

    def expanded_(lay, name):
        if _unit_lays.get(id(lay)) is lay:
            return "\n" * _unit(lay, name)["newlines"]
        return expanded(lay, name)

    def compose_(low, *a, **k):
        text = compose(low, *a, **k)
        e = _units.get((id(low.layout), low.name))
        if e is not None and text.startswith(f"{FC.FORMAT} "):
            j = text.index("\n", text.index("\n") + 1) + 1
            mark = f"CIRCUIT unit {e['newlines']}\n"
            at = j + len(mark)
            if text[j:at] != mark or text[at:at + e["newlines"]] != "\n" * e["newlines"] or not text.startswith("CIRCUIT ", at + e["newlines"]):
                raise AssertionError("compose: the unit's netlist is not where fast_stage put it")
            _streams[id(text)] = {"text": text, "at": at, "newlines": e["newlines"], "path": e["path"],
                                  "bytes": len(text) - e["newlines"] + e["bytes"]}
        _composed[:] = [text]
        return text

    def write_text_(self, data, encoding=None, errors=None, newline=None):
        if isinstance(data, str) and id(data) in _streams and _streams[id(data)]["text"] is data:
            if (encoding or "utf-8").lower().replace("_", "-") not in ("utf-8", "utf8", "locale") or newline not in (None, "\n"):
                raise ValueError("fast_stage: a streamed circuit is written as UTF-8 with \\n newlines")
            with open(self, "wb") as f:
                _of(data).update(_stream(data, f))
            return _streams[id(data)]["bytes"]
        if not isinstance(data, str) or len(data) <= CHUNK:
            return write_text(self, data, encoding, errors, newline)
        with self.open(mode="w", encoding=io.text_encoding(encoding), errors=errors, newline=newline) as f:
            for i in range(0, len(data), CHUNK):
                f.write(data[i:i + CHUNK])
        return len(data)

    forms.Circuit._new = new_
    CS.unit_lowering, IL.lower, IL.layout, FC._expanded, FC.compose = unit_lowering_, lower_, layout_, expanded_, compose_
    Path.write_text = write_text_
    IL.digest, FC.digest, FC.class_digest = sha256, digest, class_digest
    # a structural unit (`class_statement.TmplLowering`) has no netlist: its unit_sha256 stays the library's, over its template
    CS.TmplLowering.sha256_of = IL.Lowering.sha256_of
    IL.Lowering.sha256, IL.Lowering.sha256_of = property(_lowering_sha256), _lowering_sha256_of
    IL.evaluate_layout = evaluate_layout


def composed() -> str:
    """The circuit text `circuit.compose` returned last (a streamed one stands for its file)."""
    return _composed[-1]


def size(text: str) -> int:
    """The length of the circuit ``text`` stands for."""
    s = _streams.get(id(text))
    return s["bytes"] if s is not None and s["text"] is text else len(text)


def release() -> None:
    """Drop the composed text, its digests and the unit's scratch netlist."""
    for e in _units.values():
        e["path"].unlink(missing_ok=True)
    for d in (_units, _streams, _memo, _unit_lays):
        d.clear()
    _composed.clear()
