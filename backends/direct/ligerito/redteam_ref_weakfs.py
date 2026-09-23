"""red-team-ligerito: ``ref.py``'s Fiat-Shamir transcript does not bind the statement.

``ref.verify`` starts the transcript with ``absorb(root_1)``: neither the evaluation point ``z`` (the tensor ``w``) nor the
claimed ``value`` nor ``dims`` (hence |S|) is absorbed, and ``dims`` is read from the proof.  (proto ``pcs.py`` and the Rust
``pcs.rs`` absorb ``header || z || root || v``; the Rust ``refpcs.rs`` mirrors ref.py and has the same gap.)  Forgery: send
arbitrary level-1 sumcheck messages summing to a FALSE value, take the challenges, then pick ``z`` so the level-1 claim is
true of the honest fold ``v_1``; every later level is honest.  Toy radix-3 shape, two levels.

    python -m backends.direct.ligerito.redteam_ref_weakfs
"""
from __future__ import annotations

import numpy as np

from backends.direct.ligerito import ref as R

P, EXT = R.P, R.EXT


def rand_ext(rng, k=None):
    shape = (EXT,) if k is None else (k, EXT)
    return rng.integers(0, P, size=shape, dtype=np.int64)


def einv(a):
    return R.epow(a, P ** EXT - 2)


def eqv(z, r):
    one = R.ext_one()
    return R.eadd(R.emul(R.esub(one, z), R.esub(one, r)), R.emul(z, r))


def forge(state: R.ProverState, value_star: np.ndarray, rng, seed: bytes):
    dims = state.dims
    coins = R.FSCoins(seed)
    M, U, tree = state.M1, state.U1, state.tree1
    roots, levels = [tree[-1][0]], []
    coins.absorb(b"root", roots[0])
    claim, msgs, r = value_star, [], []
    for _ in range(dims.wide_log2[0]):                       # level 1: messages that merely sum to the running claim
        c1, c2 = rand_ext(rng), rand_ext(rng)
        c0 = R.escale(R.esub(R.esub(claim, c1), c2), R.INV2)
        msg = np.stack([c0, c1, c2])
        coins.absorb(b"sumcheck", R.row_bytes(msg))
        rj = coins.challenge(b"r", 1)[0]
        msgs.append(msg)
        r.append(rj)
        claim = R.eadd(c0, R.emul(rj, R.eadd(c1, R.emul(c2, rj))))
    r = np.stack(r)
    rbar = R.expand(R.eq_tensor(r))
    v = R.mat_vec(M, rbar)                                    # the honest fold
    # z AFTER the challenges: random row digits, column digits solving b(r) <ra, v> = claim
    row_rad, k = dims.row_radices(0), dims.wide_log2[0]
    z_row = rand_ext(rng, len(row_rad))
    A = R.esum(R.emul(R.expand(R.point_tensor(z_row, row_rad)), v))
    z_col = rand_ext(rng, k)
    prod = R.ext_one()
    for j in range(k - 1):
        prod = R.emul(prod, eqv(z_col[j], r[j]))
    t = R.emul(claim, einv(R.emul(A, prod)))
    rl, one = r[k - 1], R.ext_one()
    z_col[k - 1] = R.emul(R.esub(t, R.esub(one, rl)), einv(R.esub(R.escale(rl, 2), one)))
    z = np.concatenate([z_row, z_col])
    w = R.point_tensor(z, dims.radices)
    ra, ca = R.split(w, dims.tall[0])
    terms = [(R.ext_one(), ra, ca)]
    total = claim
    levels.append(R.LevelProof(np.stack(msgs), [], [], []))
    carried = [(R.emul(g, R.tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
    prev_U, prev_tree = U, tree
    for i in range(1, dims.L):                               # honest from here (prove_eval's loop body)
        M = R._reshape(v, dims.tall[i], dims.wide(i))
        U = R.encode(M, dims.n(i))
        tree = R.merkle_tree(U)
        roots.append(tree[-1][0])
        coins.absorb(b"root", roots[-1])
        prev_idx = coins.indices(b"queries", dims.n(i - 1), dims.queries[i - 1])
        prev_rows = [prev_U[s] for s in prev_idx]
        sibs = R.merkle_multi_open(prev_tree, prev_idx)
        R._absorb_opening(coins, prev_rows, sibs)
        levels[-1].opened_idx, levels[-1].opened_rows, levels[-1].siblings = prev_idx, prev_rows, sibs
        terms, total = R._batch_terms(coins, dims, i, carried, total, prev_rows, prev_idx, rbar)
        msgs_i, r, total = R._sumcheck_prove(coins, M, terms, dims.wide_log2[i])
        levels.append(R.LevelProof(msgs_i, [], [], []))
        rbar = R.expand(R.eq_tensor(r))
        v = R.mat_vec(M, rbar)
        carried = [(R.emul(g, R.tensor_eval_binary(b, r)), a) for (g, a, b) in terms]
        prev_U, prev_tree = U, tree
    coins.absorb(b"final", R.row_bytes(v))
    idx = coins.indices(b"queries", dims.n(dims.L - 1), dims.queries[dims.L - 1])
    rows = [prev_U[s] for s in idx]
    sibs = R.merkle_multi_open(prev_tree, idx)
    levels[-1].opened_idx, levels[-1].opened_rows, levels[-1].siblings = idx, rows, sibs
    return z, w, R.Proof(dims, roots, levels, v, value_star)


def main():
    rng = np.random.default_rng(1830)
    seed = b"ligerito-ref/v1"
    for L, odd3 in ((1, False), (2, True)):
        dims = R.Dims.toy(10, L, queries=8, odd3=odd3)
        f = rng.integers(0, P, size=dims.N, dtype=np.int64)
        st = R.commit(f, dims)
        value_star = rand_ext(rng)
        z, w, proof = forge(st, value_star, rng, seed)
        ok, why = R.verify(st.root, w, value_star, proof, R.FSCoins(seed))
        true_v = R.tensor_eval(f, w)
        print(f"L={L} radix3={odd3} dims={dims}: ref.verify -> {ok} ({why}); claimed value != <f, w(z)>: "
              f"{not np.array_equal(true_v % P, value_star % P)}")
    # dims come from the proof: an honest proof at |S| = 1 per level verifies (no floor on the queries)
    dims1 = R.Dims.toy(10, 2, queries=1)
    f = rng.integers(0, P, size=dims1.N, dtype=np.int64)
    st = R.commit(f, dims1)
    z = rand_ext(rng, len(dims1.radices))
    w = R.point_tensor(z, dims1.radices)
    pf = R.prove_eval(st, w, R.FSCoins(seed))
    print(f"|S| = 1 per level (prover-chosen dims): ref.verify -> {R.verify(st.root, w, pf.value, pf, R.FSCoins(seed))[0]}")


if __name__ == "__main__":
    main()
