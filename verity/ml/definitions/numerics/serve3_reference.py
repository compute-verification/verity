"""Vectorized reference evaluator of `Serve@3`'s padding layer — the inactive transition and its outputs
(lane PAD-IMPL; semantics: PAD-SPEC v1 §4–§8).

The padded circuit adds one semantic layer over the `Serve_v2` skeleton: the flag chain, the guards, and
the row muxes.  Everything in that layer is a word computation on the raw samples `y_0..y_{D_MAX}` (the
`TokenSelect` outputs, pre-mux) and the appended cache rows, so it vectorizes exactly (numpy `uint32`
words; the pattern of `rmsnorm_fused.py`: a numpy twin checked bit-for-bit against the gate-by-gate
circuit evaluation — `tests/program/test_serve3.py::test_reference_evaluator_matches_the_gates`):

* `pad_chain(y, eos)` — the flag chain and the token slots (spec §4, §14.2).  `live[g] = live_{g+1} =
  ∏_{j ≤ g} [y_j ≠ eos]` (the cumulative product *is* the absorbing `BitAnd` chain: once a factor is 0 the
  product never rises), `out[g] = y_g if live_{g} else 0` (`PAD_TOKEN_ID = 0`, spec §6).  Total on words:
  in the inactive suffix `y_g` is canonical garbage and the product absorbs whatever the stop test returns.
* `mux_rows(live, rows)` — the row muxes (spec §6/§7): appended k/v words `where(live_s, computed, 0x0000)`
  — the zero row, vectorized over steps and words.
* `interpret(out, live)` — the interpretation map (spec §8): `n = Σ live[0..D_MAX-1]`, `transcript =
  out[0..n-1]` (EOS-exclusive; slot `n` is stop evidence when `n < D_MAX`, never a semantic output; the
  suffix equals the pad constant and is not an output).
* `inactive_boundary(C)` — the step-boundary relation of spec §6, in its trivial form: for
  `live_in = 0` the `PadDecodeStep_v1` boundary outputs are the constants `(tok_out, live_out, k_new,
  v_new) = (0, 0, 0…0, 0…0)`, checkable without touching an interior gate.

What this module does **not** vectorize: the inactive step *bodies* ("canonical garbage", spec §6) — their
values are the registered gate semantics on the actual inputs (tensor-core k-chains, FA2 online softmax,
MUFU tables), whose bit-exact reference is the circuit evaluator itself; `materialize(prog, inputs)` runs it
and returns the full transcript, which is how a prover materializes `canonical, not captured` leaves
(spec §12).  Vectorized twins of those kernels are the numerics lane's C++ model, not a padding concern.
"""

from __future__ import annotations

import numpy as np

__all__ = ["pad_chain", "mux_rows", "interpret", "inactive_boundary", "materialize"]


def pad_chain(y: np.ndarray | list[int], eos: int) -> tuple[np.ndarray, np.ndarray]:
    """The guard chain over all `D_MAX+1` sampling events, vectorized.  `y[g]` = event `g`'s raw sample
    (event 0 = prefill's `Final`; in the inactive suffix, canonical garbage).  Returns `(out, live)` —
    the circuit's committed boundary (spec §8): `out[g]` the muxed token slot, `live[g]` the flag *after*
    event `g`."""
    y = np.asarray(y, dtype=np.uint64)
    stop = (y == np.uint64(eos)).astype(np.uint8)
    live_after = np.cumprod(1 - stop, dtype=np.uint8)  # live_{g+1} = BitAnd(live_g, BitNot(stop_g)), absorbing
    live_before = np.concatenate(([np.uint8(1)], live_after[:-1]))  # live_0 = Const1[0x1]
    out = np.where(live_before.astype(bool), y, np.uint64(0)).astype(np.uint64)  # SelectI32(live_g, y_g, 0)
    return out.astype(np.uint32), live_after


def mux_rows(live_before_step: np.ndarray | list[int], rows: np.ndarray) -> np.ndarray:
    """The row muxes of all decode steps, vectorized: `rows[s-1]` = step `s`'s computed k/v words (any
    trailing shape, uint16 bf16 words), `live_before_step[s-1] = live_s`.  Appended words are
    `SelectBf16(live_s, computed, 0x0000)` (spec §6: the zero row)."""
    live = np.asarray(live_before_step, dtype=bool).reshape((-1,) + (1,) * (np.asarray(rows).ndim - 1))
    return np.where(live, np.asarray(rows, dtype=np.uint16), np.uint16(0))


def interpret(out: np.ndarray | list[int], live: np.ndarray | list[int]) -> tuple[int, tuple[int, ...]]:
    """The interpretation map (spec §8), reading only committed circuit outputs: `(n, y_0..y_{n-1})` with
    `n = Σ live[0..D_MAX-1]` and EOS-exclusive transcript `out[0..n-1]`."""
    out, live = np.asarray(out), np.asarray(live)
    n = int(live[:-1].sum())
    return n, tuple(int(x) for x in out[:n])


def inactive_boundary(C: dict) -> tuple[int, int, np.ndarray, np.ndarray]:
    """The inactive step-boundary constants of spec §6: for `live_in = 0`, `PadDecodeStep_v1`'s boundary is
    `(tok_out, live_out, k_new, v_new) = (0, 0, 0…0, 0…0)` — `k_new`/`v_new` as `[NL][KVH][D]` zero words."""
    z = np.zeros((C["NL"], C["KVH"], C["D"]), dtype=np.uint16)
    return 0, 0, z, z.copy()


def materialize(prog, inputs: list[list[int]]) -> dict[int, int]:
    """Materialize *every* circuit value (live and `canonical, not captured` alike, spec §12) from the
    committed inputs by the registered gate semantics: the full transcript `{root gate index: word}` of
    `prog` on `inputs` (one flat word list per program input, leaf order)."""
    from verity.ml.circuits.evaluate import evaluate_region
    from verity.ml.circuits.layout import resolve
    circ = prog.circuit
    t: dict[int, int] = {}
    for pi, vals in enumerate(inputs):
        coll = prog.input_coll(pi)
        if coll.type.leaves != len(vals):
            raise ValueError(f"input {pi}: expected {coll.type.leaves} words, got {len(vals)}")
        for j, v in enumerate(vals):
            t[resolve(circ.root_scope, coll.refs[j])] = v
    evaluate_region(circ, circ[0:prog.gates], t)
    return t
