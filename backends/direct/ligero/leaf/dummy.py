"""``dummy`` -- a TEST-ONLY leaf scheme: a linear (Horner) running sum of the row's words in four BabyBear lanes plus
one quadratic hint row, so a second scheme exercises every seam of the interface (registry, ``compose``, the linked
carry rows, an EXTRA hint row placed through ``chain_witness``'s third result, the digest pins, tree building, statement
v5 with a ``+dummy`` relation suffix, the negatives) WITHOUT touching ``hashchain.py``.

INSECURE by construction: the state is linear in the words, so collisions are a one-line linear solve.  The registry flags
it ``test_only``; the bench runner refuses it; the Rust verifier does not know it (a ``+dummy`` statement is refused
there -- which is itself the test that the verifier checks the scheme name).

Definition (``P`` the BabyBear prime, ``G_i`` four fixed generators, ``role`` the operand role):
    s_i <- s_i * G_i + w          for every word w of the row in order, i = 0..3, starting from s = 0
    t = s_0 * (s_1 + role + 1)                                                (the quadratic hint row)
    digest = (t, s_1, s_2, s_3);  carried state = (s_0, s_1, s_2, s_3)
The column gadget applies the k-word Horner step to the carried state; the split of the row into columns does not change
the result, so :meth:`native` needs no column width.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ..compile import Expr, LigeroCtx, P, Quadratic
from . import registry

__all__ = ["DummyLeaf", "GENS", "horner_np"]

GENS = (3, 5, 7, 11)            # the four lanes' Horner multipliers
DIGEST_ELEMS = 4
CARRY_ELEMS = 4
SCHEMA = "dummy-linear4/row/test"
PARAMS_SHA256 = hashlib.sha256(f"dummy-linear4|gens={GENS}|p={P}".encode()).hexdigest()


def _reduce(e):
    if isinstance(e, Expr):
        return Expr(e.ctx, {k: v % P for k, v in e.lin.items()}, e.c % P)
    return e % P


def horner_np(rows: np.ndarray) -> np.ndarray:
    """(R, K) words -> (R, 4) lanes ``sum_t w_t G_i^(K-1-t) mod P`` (numpy int64; every partial product < 2^62)."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.zeros((rows.shape[0], CARRY_ELEMS), dtype=np.int64)
    for i, g in enumerate(GENS):
        s = np.zeros(rows.shape[0], dtype=np.int64)
        for t in range(rows.shape[1]):
            s = (s * g + rows[:, t]) % P
        out[:, i] = s
    return out


def _digest_of(state: np.ndarray, role: int) -> np.ndarray:
    d = state.copy()
    d[:, 0] = (state[:, 0] * ((state[:, 1] + role + 1) % P)) % P
    return d


class DummyLeaf:
    name = "dummy"
    assumption = "NONE (test-only linear running sum: collisions by a linear solve)"
    digest_elems = DIGEST_ELEMS
    carry_elems = CARRY_ELEMS
    schema = SCHEMA
    params_sha256 = PARAMS_SHA256
    hash_name = "dummy-linear4"
    privacy_note = "linear, unsalted, test-only: binding-only and not even that (collisions by a linear solve)"

    # --- committer -----------------------------------------------------------------------------------------------------
    def native(self, rows: np.ndarray, *, word_bits: int, role: int) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 2 or ((rows < 0) | (rows >= 2**word_bits)).any():
            raise ValueError(f"dummy: rows must be (R, K) words of {word_bits} bits")
        return _digest_of(horner_np(rows), role)

    def leaf_bytes(self, digest_row) -> bytes:
        d = [int(x) for x in digest_row]
        if len(d) != DIGEST_ELEMS or any(not 0 <= x < P for x in d):
            raise ValueError("dummy: a digest is 4 BabyBear elements")
        return b"".join(x.to_bytes(4, "little") for x in d)        # little-endian on purpose: not Poseidon2's framing

    # --- gadget ----------------------------------------------------------------------------------------------------------
    def gadget(self, ctx: LigeroCtx, words: Sequence[Expr], *, word_bits: int, role: int, row_words: int,
               carry_in: Sequence[Expr], first: bool = False, last: bool = True) -> tuple[list[Expr], list[Expr]]:
        if len(carry_in) != CARRY_ELEMS:
            raise ValueError("dummy: the carried state is 4 lanes")
        k = len(words)
        state = []
        for i, g in enumerate(GENS):
            e = carry_in[i] * pow(g, k, P)
            for j, w in enumerate(words):
                e = e + w * pow(g, k - 1 - j, P)
            state.append(_reduce(e))
        # the extra hint row: t = s0 * (s1 + role + 1), placed by chain_witness's third result under this exact name
        t = ctx.hint(f"dummy.t.role{role}")
        ctx.sys.quadratic.append(Quadratic(f"dummy.t.role{role}", state[0], _reduce(state[1] + role + 1), t))
        return [t, state[1], state[2], state[3]], list(state)

    def witness(self):
        return None

    # --- chain witness ---------------------------------------------------------------------------------------------------
    def chain_witness(self, words, *, word_bits: int, role: int, row_words: int, n_vus: int, steps: int, l: int):
        """``words`` (l, k) int64 torch tensor -> (carry_in (4, l), digests (n_vus, 4), {hint name: (l,)})."""
        import torch
        dev = words.device
        k = words.shape[1]
        carry_in = torch.zeros((CARRY_ELEMS, l), dtype=torch.int64, device=dev)
        t_row = torch.zeros(l, dtype=torch.int64, device=dev)
        st = torch.zeros((CARRY_ELEMS, n_vus), dtype=torch.int64, device=dev)
        cols0 = torch.arange(n_vus, device=dev) * steps
        for s in range(steps):
            cols = cols0 + s
            carry_in[:, cols] = st
            w = words[cols].T                                   # (k, n_vus)
            for i, g in enumerate(GENS):
                acc = st[i]
                for j in range(k):
                    acc = (acc * g + w[j]) % P
                st = st.clone()
                st[i] = acc
            t_row[cols] = (st[0] * ((st[1] + role + 1) % P)) % P
        digests = st.T.clone()
        digests[:, 0] = t_row[cols0 + steps - 1]
        return carry_in, digests, {f"dummy.t.role{role}": t_row}

    def census(self) -> dict:
        return {"dummy_lanes": CARRY_ELEMS}


registry.register(DummyLeaf, test_only=True)
