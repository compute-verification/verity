"""Several repeated unit types in one ``verity/flock-circuit`` block (a prototype; the Lean verifier refuses the form until its
proof covers it): a template split into a repeated **k-step unit** (``units_per_vu`` per VU, reading the rows' leaves) and a
repeated **stage** (a second unit type, ``count`` slots per VU, reading the units' cut words), joined by Δ's wires, the
stage returning the outputs. `circuit.compose` takes the split's tail as given (``tail``): ``per_vu`` holds the stage's count,
the wires are slot-indexed, and ``out_net_ports`` names each output word's ``[slot, port]``. Neither ``flock-ir-unit/v2`` nor
the block matrix ``A = I ⊗ A_0 + Δ`` changes: a slot type is folded once by the lincheck however many slots it has, so the
verifier's work is the sum of the distinct types' nonzeros.

:func:`fp8_block`: ``ScaledMmFp8Block_v2{K,N,G}`` on one token row (DeepGEMM's and CUTLASS's block-scaled FP8 linear), unit
``(n, kb)`` the tile's ``G/32`` k32 steps from +0 and ``FMUL(sx[kb], ws[n // G][kb])``, stage slot ``n`` the ``FFMA`` chain
over the ``K/G`` tiles from +0 and the bf16 cast: ``_block_acc``'s own order, checked against the Definition's IR reference
and the served kernel's numpy twin by the caller. :func:`fp8_block_linked`: the same two circuits as two statements whose
units return their outputs (the writer's output row of the cut words, the reader's input row), the form a link between two
statements' rows (PROTOCOL.md §16.13) proves."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import circuit as FC
from . import gf2
from . import ir_lower as IL
from . import tail as TL


@dataclass
class Split:
    """A template as k-step units and repeated stages: ``low`` the units (their circuit, layout and leaf maps), ``tail`` the
    stages (``None``: the units return the outputs), ``rows`` the input rows ``(name, words)``, ``out_words`` the output
    words, ``definition`` the Definition it computes (its Program digest is META's), ``forged`` the same stages computing
    another function (the selftest's ``tail_swapped``)."""

    name: str
    low: IL.Lowering
    tail: TL.Tail | None
    rows: list
    out_words: int
    definition: object = None
    forged: TL.Tail | None = None


class _SplitLowering(IL.Lowering):
    @property
    def cut(self) -> dict | None:
        U, lay = self.units, self.layout
        if not U.cut:
            return None
        return {"words": len(U.cut), "split": self.name,
                "in": [[[lay.in_port_cols[j], p] for j, (k, p) in enumerate(src) if k == "cut"] for src in U.in_src],
                "out": [[[lay.out_port_cols[j] - lay.out_col * IL.WORD, p] for j, (k, p) in enumerate(src) if k == "cut"]
                        for src in U.out_src]}


def _call(C, d, *words):
    """``d`` (a bound Boolean Definition, every leaf one bit) on bit lists ``words``: its output bits."""
    return [b for o in IL.lower(C, d, [[b] for w in words for b in w]) for b in o]


def _lowering(name: str, C, n_in: int, out_widths: list, in_src: list, out_src: list, rows: list, out_words: int, cut: int = 0,
              rets: int = 0) -> IL.Lowering:
    from verity.ml.circuits.types import Array, Tuple, Value

    units = IL.Units(target=None, in_widths=[16] * n_in, out_widths=out_widths, in_src=in_src, out_src=out_src, signature=(name,),
                     gates=[], arg_gates=[], ret_gates=[], cut=list(range(cut)), tail_rets=list(range(rets)))
    lay = IL.layout(C, [f"in{j}" for j in range(n_in)], [f"out{j}" for j in range(len(out_widths))])
    sig = (tuple((r, Array(w, Value(16))) for r, w in rows), Tuple(Array(out_words, Value(16))))
    return _SplitLowering(name, SimpleNamespace(id=name, signature=lambda: sig), units, C, lay)


def _fp8_pieces():
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG
    from verity.ml.circuits.ops.boolean import scalar as SC

    return bind(BG.HopperE4m3QgmmaDot32), bind(SC.F32Mul), bind(SC.F32Fma), bind(BG.F2fpBf16)


def _kstep_circuit(G: int, word_outs: bool) -> gf2.Circuit:
    """The k-step unit: inputs the tile's G e4m3 x bytes, sx[kb], the weight column's G bytes and its block's scale (u16 words,
    low byte first); outputs ``temp`` (G/32 k32 steps from +0) and ``s = FMUL(sx, sw)``, as two 32-bit ports (``word_outs``:
    four 16-bit ports, low halves first)."""
    dot, mul, _fma, _cast = _fp8_pieces()
    tile = G // 2
    C = gf2.Circuit()
    ins = [C.input(f"in{j}", 16) for j in range(2 * tile + 4)]
    bits = lambda a, b: [x for w in ins[a:b] for x in w]  # noqa: E731
    xb, sxb, wb, swb = bits(0, tile), bits(tile, tile + 2), bits(tile + 2, 2 * tile + 2), bits(2 * tile + 2, 2 * tile + 4)
    temp = [gf2.ZERO] * 32
    for st in range(G // 32):
        temp = _call(C, dot, temp, xb[256 * st:256 * (st + 1)], wb[256 * st:256 * (st + 1)])
    s = _call(C, mul, sxb, swb)
    outs = [temp[:16], temp[16:], s[:16], s[16:]] if word_outs else [temp, s]
    for j, o in enumerate(outs):
        C.output(f"out{j}", o)
    return C


def _scale_circuit(T: int, word_ins: bool, reverse: bool = False) -> gf2.Circuit:
    """The scale stage: inputs ``(temp_kb, s_kb)`` per tile (``word_ins``: each as two 16-bit words), ``acc = FFMA(temp, s, acc)``
    from +0 over the T tiles (``reverse``: last tile first, the forged stage: the same gates and output column, rewired),
    output the bf16 cast (one 16-bit port)."""
    _dot, _mul, fma, cast = _fp8_pieces()
    C = gf2.Circuit()
    if word_ins:
        w = [C.input(f"in{j}", 16) for j in range(4 * T)]
        ins = [w[2 * j] + w[2 * j + 1] for j in range(2 * T)]
    else:
        ins = [C.input(f"in{j}", 32) for j in range(2 * T)]
    acc = [gf2.ZERO] * 32
    for kb in reversed(range(T)) if reverse else range(T):
        acc = _call(C, fma, ins[2 * kb], ins[2 * kb + 1], acc)
    C.output("out0", _call(C, cast, acc))
    return C


def _fp8_shape(K: int, N: int, G: int):
    """``(T, NB, x words, w words)``: the K/G tiles, the ``cdiv(N, G)`` weight-scale rows (`registry.fp8.weight_blocks`)."""
    if K % G or G % 32:
        raise ValueError(f"ScaledMmFp8Block: K={K} a multiple of G={G}, G a multiple of 32")
    T, NB = K // G, -(-N // G)
    wx, ww = K // 2 + 2 * T, N * K // 2 + 2 * T * NB
    return T, NB, wx, ww


def _kstep_leaves(K: int, N: int, G: int, n: int, kb: int) -> list[int]:
    """Unit ``(n, kb)``'s 2G/2 + 4 leaves: the x row's tile and sx[kb], the w row's column n tile and ws[n // G][kb]."""
    T, _NB, wx, _ww = _fp8_shape(K, N, G)
    tile = G // 2
    x = [kb * tile + i for i in range(tile)] + [K // 2 + 2 * kb + i for i in range(2)]
    w = [wx + n * K // 2 + kb * tile + i for i in range(tile)] + [wx + N * K // 2 + 2 * (T * (n // G) + kb) + i for i in range(2)]
    return x + w


def fp8_block(K: int, N: int, G: int = 128, definition: object = None) -> Split:
    """``ScaledMmFp8Block_v2{K,N,G}`` as ``N * K/G`` k-step units (unit ``u = n T + kb`` returns cut words ``2u`` = temp and
    ``2u + 1`` = s) and a scale stage of ``N`` slots (slot ``n`` reads its T tiles' pairs, returns output word ``n``). Rows
    ``x`` = xq || sx, ``w`` = w[0..N] || ws, little-endian bit concatenations (`class_statement.unit_lowering`'s).
    ``definition`` is the bound Definition whose Program digest META names (the caller's: it is an integration's)."""
    T, _NB, wx, ww = _fp8_shape(K, N, G)
    rows = [("x", wx), ("w", ww)]
    in_src = [[("leaf", p) for p in _kstep_leaves(K, N, G, n, kb)] for n in range(N) for kb in range(T)]
    out_src = [[("cut", 2 * u), ("cut", 2 * u + 1)] for u in range(N * T)]
    name = f"multi-unit/ScaledMmFp8Block_v2{{K={K},N={N},G={G}}}"
    low = _lowering(f"{name}/kstep", _kstep_circuit(G, False), G + 4, [32, 32], in_src, out_src, rows, N, cut=2 * N * T, rets=N)
    slot_ins = [[("cut", 2 * (n * T + kb) + j) for kb in range(T) for j in range(2)] for n in range(N)]
    slot_outs = [[("ret", n)] for n in range(N)]
    tail = lambda S: TL.Tail([TL.Stage(S, slot_ins[0], slot_outs[0], [32] * (2 * T), [16], count=N, slot_ins=slot_ins,  # noqa: E731
                                       slot_outs=slot_outs)], [], [], list(range(N)))
    return Split(name, low, tail(_scale_circuit(T, False)), rows, N, definition,
                 forged=tail(_scale_circuit(T, False, reverse=True)))


def fp8_block_linked(K: int, N: int, G: int = 128) -> tuple[Split, Split]:
    """The same split as two statements whose units return their outputs: the writer's units ``(n, kb)`` return
    ``temp, s`` as four u16 words (output word ``4u + i``; padded with zero-leaf units to a power of two, as a statement of
    returned outputs needs), its output row ``c``; the reader's ``N`` units read ``c``'s words ``4 n T ..`` and return the
    coordinates."""
    T, _NB, wx, ww = _fp8_shape(K, N, G)
    U = 1 << (N * T - 1).bit_length()
    in_src = [[("leaf", p) for p in _kstep_leaves(K, N, G, n, kb)] for n in range(N) for kb in range(T)]
    in_src += [[("zero", 0)] * (G + 4) for _ in range(U - N * T)]
    out_src = [[("ret", 4 * u + i) for i in range(4)] for u in range(U)]
    name = f"multi-unit/ScaledMmFp8Block_v2{{K={K},N={N},G={G}}}"
    writer = Split(f"{name}/writer", _lowering(f"{name}/writer", _kstep_circuit(G, True), G + 4, [16] * 4, in_src, out_src,
                                                [("x", wx), ("w", ww)], 4 * U), None, [("x", wx), ("w", ww)], 4 * U)
    r_src = [[("leaf", 4 * n * T + j) for j in range(4 * T)] for n in range(N)]
    r_out = [[("ret", n)] for n in range(N)]
    reader = Split(f"{name}/reader", _lowering(f"{name}/reader", _scale_circuit(T, True), 4 * T, [16], r_src, r_out,
                                                [("c", 4 * U)], N), None, [("c", 4 * U)], N)
    return writer, reader


def evaluate(split: Split, flat: np.ndarray) -> tuple[np.ndarray, bool]:
    """``(outputs (n, out_words), every row satisfied)``: the laid-out units over every instance, then each stage's slots on
    the cut words they read."""
    flat = np.asarray(flat, dtype=np.uint64)
    n = flat.shape[0]
    res, cuts, ok = split.low.evaluate(flat)
    good = int(ok) == (1 << (n * split.low.units.n)) - 1
    if split.tail is None:
        return res.astype(np.uint16), good
    out = np.zeros((n, split.out_words), dtype=np.uint16)
    for st in split.tail.stages:
        idx = np.asarray([[k for _kind, k in st.tags(s)[0]] for s in range(st.count)], dtype=np.int64)
        ports = [cuts[:, idx[:, j]].reshape(-1) for j in range(idx.shape[1])]
        outs, ok2 = IL.evaluate_layout(TL.stage_layout(st), ports, n * st.count)
        good &= int(ok2) == (1 << (n * st.count)) - 1
        for s in range(st.count):
            for j, tag in enumerate(st.tags(s)[1]):
                o = outs[j].reshape(n, st.count)[:, s]
                if tag[0] == "ret":
                    out[:, tag[1]] = o
                else:
                    cuts[:, tag[1]] = o
    return out, good


def _sub(split: Split):
    from verity.ml.circuits.codec import encode_program, program_digest
    from verity.ml.circuits.program import Program

    port = lambda nm: SimpleNamespace(name=nm, dtype="u16")  # noqa: E731
    d = split.definition

    def ev(ins):
        flat = np.concatenate([np.asarray(ins[r][0], dtype=np.uint64) for r, _w in split.rows])[None, :]
        return {"y": [evaluate(split, flat)[0][0]]}

    digests = (lambda _s: {d.id: program_digest(encode_program(Program(d)))}) if d is not None else \
        (lambda _s: {split.name: hashlib.sha256(split.name.encode()).hexdigest()})
    return SimpleNamespace(id=split.name, template=SimpleNamespace(id="multi-unit"), inputs=[port(r) for r, _w in split.rows],
                           outputs=[port("y")], program_digests=digests, evaluate=ev,
                           **({"definition": lambda **_s: d} if d is not None else {}))


def counts(split: Split) -> dict:
    """Each slot type's ANDs and A, B nonzeros (the verifier folds each type once) and slots per VU."""
    def one(lay, per_vu):
        c = lay.counts
        return {"ands": c["ands"], "nnz": c["nnz_a"] + c["nnz_b"], "useful": lay.useful, "per_vu": per_vu}
    out = {"unit": one(split.low.layout, split.low.units.n)}
    for i, st in enumerate(split.tail.stages if split.tail else []):
        out[f"stage{i}"] = one(TL.stage_layout(st), st.count)
    return out


def stage(split: Split, rows: dict, out_dir: Path, salt_key: bytes, want: np.ndarray | None = None) -> dict:
    """``rows`` (``{row: (n, words)}`` u16) as the split's statement under ``out_dir``, every row a ``sha512/row/v2`` bit row of
    exactly its words: ``circuit.txt``, ``inst-<n>.bin``, ``pub-<n>.bin``, ``outs.npy`` (the outputs, which ``want`` must equal
    where given), and ``circuit-forged.txt`` for a split with forged stages. Returns META's shape and the counts."""
    sub = _sub(split)
    n = len(rows[split.rows[0][0]])
    kw = {"program": None if split.definition is not None else "0" * 128, "port_bits": [16 * w for _r, w in split.rows],
          "v2_out_rows": (("y",),)}
    text = FC.compose(split.low, sub, tail=split.tail, **kw)
    flat = np.concatenate([np.asarray(rows[r], dtype=np.uint64) for r, _w in split.rows], axis=1)
    outs, ok = evaluate(split, flat)
    if not ok:
        raise AssertionError(f"{split.name}: a slot is unsatisfied on its rows")
    if want is not None and not np.array_equal(outs, np.asarray(want, dtype=np.uint16)):
        raise AssertionError(f"{split.name}: the outputs are not the reference's")
    content = hashlib.sha256(json.dumps([split.name, n]).encode()).hexdigest()
    iset = SimpleNamespace(subcircuit=sub, name=split.name.replace("/", "-"), content_digest=content)
    meta = json.loads(text.split("\n", 2)[1][5:])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "circuit.txt").write_text(text)
    if split.forged is not None:
        (out_dir / "circuit-forged.txt").write_text(FC.compose(split.low, sub, tail=split.forged, **kw))
    FC.write(text, iset, 0, n, out_dir / f"inst-{n}.bin", out_dir / f"pub-{n}.bin", salt_key,
             rows={r: np.asarray(rows[r], dtype=np.uint16) for r, _w in split.rows}, outs={"y": outs})
    np.save(out_dir / "outs.npy", outs)
    return {"name": split.name, "n": n, "k_log": meta["k_log"], "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"],
            "ranges": meta["ranges"], "counts": counts(split), "circuit_sha512": FC.digest(text), "circuit_bytes": len(text)}
