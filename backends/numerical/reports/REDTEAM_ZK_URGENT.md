# URGENT (red-team-zk, 2026-09-22 ~10:30Z): B-Ligero `COMPLETE_ZK_BACKEND` must be downgraded

Three findings against the b-zk claim (run `r20260922-092404-2a1b`, 4096 VUs as 25 x 170, `--zk`, HM96 coins).
Full report: `redteam_zk.md` / `.json`; code `backends/direct/ligero/redteam/`.

## 1. ACCEPTED-LEAK: the chain message `q` is unmasked on H -- every intermediate accumulator is in the transcript

`q_d = Q_d + Z_H s_chain_d` (protocol.py "ZK masking", PROTOCOL.md 8b "chain mask Z_H s_chain (sum over H = 0)").
`Z_H s_chain` vanishes at every point of `H`, not merely in its sum, so `q_d|_H = Q_d|_H = (sum_i coef[d,i,j] W[rows_i][j])_j`
exactly -- the per-column share of the cross-column constraints, a witness-dependent linear function the verifier
reads with `_fold_to_H(q, l)`. For the honest witness, `Q_d(h_j) = sum_x rl_x[j-1] c_x[j] - sum_x rl_x[j] c_x[j+1] + (public)`
with the verifier's own coefficients `rl`; `c[start] = 0` is public, so walking each VU forward gives `D = 6`
equations in the 3 components of the next accumulator at every step. **Demonstrated** (`redteam/q_leak.py`): all
`95 x n_vus` accumulators of every VU recovered exactly from `(q, root, coins, statement)` -- 190/190 at `l = 256`
on the laptop, 16150/16150 at the measured layout `l = 16384` on vy-g3 (see `redteam_zk.json`). Mask dimension
count: admissible space of `q` has dimension `l + k - 2`, the mask spans `k - 1`: deficiency `l - 1 = 16383` field
elements per coordinate, `6 x 16383` per proof in the clear (`redteam/mask_rank.py`).

This holds for the HONEST verifier: it is an HVZK failure of the 8b construction, not only a malicious-verifier one.
Under the prover's current statement the operands `a, b` and the words `y16` are PUBLIC (`statement_digest`), so the
accumulators are computable from the statement anyway and nothing secret is lost *today*; but the campaign's ZK
target (`ZK_CONSTRUCTION.md` 0: operands, accumulators, outputs, hints are private) makes this a total break of the
accumulator chain (and, through consecutive accumulators, of the block dot products of the operands) the moment the
operands are committed instead of published. The opened columns, `w`, `v`, `h` are correctly masked (exact rank
computation + per-position statistics at the real layout: no leak).

Fix (owner b-zk, one line): mask `q` with a uniform polynomial of degree `< l + k - 1` whose sum over `H` is zero
(AHIV22's `u_add`: `k + l - 2` free coefficients), i.e. add `u - (sum_H u)/l` for uniform `u`, and open its row like
the others (the verifier's column check becomes `q(eta_c) = sum_i R_i(eta_c) U_i[c] + u(eta_c)`), instead of `Z_H s_chain`.

## 2. The 8c simulator does not exist as written; it exists only in the programmable random-oracle model

PROTOCOL.md 8c: "knowing `(r1, r2)` and hence `r, rho, cols`". False: `r, rho, rc = SHAKE(H(stmt, c, root, r1))` and
`cols = H(stmt, c, root, w, h, q, v, r2)` -- the coins are hashes of the prover's own messages together with `r_i`.
After the rewind the simulator knows `r1` but not `r` (it depends on the `root` it has yet to build), and the
column tests need committed rows that fit challenges derived from that root at columns that are a hash of the
messages. `redteam/simulator.py` implements the simulator against the real verifier: it works only by programming
`H(... root ... r1) := seed1`, `H(... root, w, h, q, v ... r2) := seed2` (the same transcript is rejected -- "column
challenge mismatch" -- as soon as the oracle is not programmed). So malicious-verifier ZK of the implemented protocol
is a random-oracle statement, the assumption `ZK_CONSTRUCTION.md` 3.2/3.3 chose step 0 to avoid; the label must
carry it. (The commitment is still needed in the ROM: without it V* chooses `r1` after `root` and evaluates the
unprogrammable hash on `(root, r1')` for many `r1'`.)

## 3. Deployment model: the measured mode is non-interactive, where the HM96 step is a no-op and 2^-128 is not the bound

`bench-vu` runs `prove(coins=None)`: the prover samples `r1, r2` itself and commits to them -- i.e. Fiat-Shamir with a
prover-chosen nonce (`_challenge1` includes `root`, so the challenge is not predictable before commitment, but the
prover controls the nonce: unbounded restarts). The reported `2^-128.05` is the interactive statistical bound
(`accounting.py` docstring: "live verifier challenges, no Fiat-Shamir"); in the non-interactive mode the campaign's
own rule (`ZK_CONSTRUCTION.md` 3.3, `zk_cost.fiat_shamir_reparameterisation`) requires `q x eps` accounting
(`t ~ 279`, no longer fitting `t_pad = 256`). The ZK claim (interactive V*) and the soundness claim as measured
(non-interactive) are in different models; the code supports the interactive mode (`verify(coins=own)`) at the
same arithmetic cost, so this is a labelling repair, but the ledger rows as written mix the two.

## Verdict on the label

**DOWNGRADE below `COMPLETE_HVZK_BACKEND`.** As a ZK construction with a committed-private witness the 8b masking
leaks `q|_H` (finding 1) to the honest verifier, so it is not HVZK; as a protocol for the *current* statement it is
trivially ZK (the witness is a function of the public statement), so no privacy label is meaningful. Honest label
for `r20260922-092404-2a1b`: `NON_ZK_PROOF_DIAGNOSTIC` with note "8b masking of columns/w/v/h verified; q unmasked
on H (accumulators in the clear); operands and outputs public in the statement; interactive soundness bound quoted
for a non-interactive run". After the one-line fix of finding 1 and a private-operand statement: `COMPLETE_HVZK_BACKEND`;
`COMPLETE_ZK_BACKEND (random-oracle)` with finding 2 written into PROTOCOL.md 8c, or a challenge derivation
`r := PRG(r1)`, `cols := PRG(r2)` (coins are the verifier's messages alone) for CRH-only malicious-verifier ZK --
which then forbids the non-interactive reading entirely.
