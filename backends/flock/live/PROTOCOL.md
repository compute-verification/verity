# verity/flock-circuit: the masked protocol (M1) and its coin commitment (M2)

What `flock-circuit --zk` sends, where every mask comes from, and what the verifier checks. This is the layout the Lean
reference prover must reproduce byte for byte (soundness DESIGN.md §6: a deterministic function of statement, witness,
prover seed and verifier coins), and the statement its zero-knowledge theorem is about.

Code:
- `src/zk_veil.rs`: masking, replay, constraints, the inner proof, the simulator's pieces.
- `src/bin/flock-circuit.rs`: `prove_cpu_zk`, `zk_constraints`, `verify_zk_with`, `simulate_full`, `zkstat`, `zkaudit`; M2's
  `VStar`, `gk_simulate`, `rewind`, `zkrewind`.
- `src/coin_tree.rs` (M2): the verifier's coin commitment and the prover's check of every opening.
- `../flock-zk-b684b12.patch`: Ligerito level 0's hiding in upstream Flock (`pcs::ligerito::zk`, `commit::commit_zk`, the F256
  split prover and verifier).

**Status.** CPU prototype, one template (RoPE d64), lane flock-zk. Every message a session sends is masked, committed or
public, and the audit (`zkaudit`) attributes every byte. The target notion is VEIL's special honest-verifier zero
knowledge for the whole session (both repetitions), meaning one simulator for every fixed coin sequence (DESIGN.md §11),
statistical in the common-reference-string model under the assumptions of §7. Malicious-verifier zero knowledge is M2's
goal, from its coin commitment and single rewind (§9). It does not hold yet: a verifier that commits a zerocheck outer coin
`r_j = 1` gets an answer that depends on the witness, until the prover refuses that coin. The first red-team review
(red-team-m1-zk) found that the two reps shared level 0's extra lanes; §7 now gives each rep its own pair, and the
re-review grants M1 with conditions. The proof class
stays `NON_ZK_PROOF` until the Lean proof covers it.

Conventions. The field is GF(2^128) (`F128`, characteristic 2, so `+` is `−`). F256 is `F128 ⊕ F128·U`, written by
coordinates `(c0, c1)`. An F128's bytes are `lo` then `hi`, each little-endian. `v'` is a sent (masked) value, `v` the
value upstream's prover computed, `h` a pad, and `v' = v + h`. `m` is the dense variable count and `k_log` the block size
(RoPE: `m = 25..27`, `k_log = 22`). Ligerito at `m = 25`: `initial_k = 6` (64 lanes), `L = 2^12` rows per lane, rates
`1/2, 1/4, 1/8, 1/16`, queries `277, 106, 71, 53` (level 0's count is §7's, above Fast100's 218), OOD samples `1, 2, 2, 2`.

## 1. Order of one repetition

Everything below is on the rep's live stream (`circuit/rep{r}`), in this order. Items marked M1 are new; the rest is
M0 and upstream Flock.

~~~text
bind_statement      label flock-r1cs-v0 · bytes statement digest · bytes witness cap (68-lane level-0 tree)
zerocheck           label flock-zerocheck-v0
  M1                label verity-flock-zk-pads-v0 · bytes pads root (64)
                    squeeze r_skip (6) · squeeze r_outer (m-13)
                    slice round1_ab' (64) · slice round1_c' (64) · squeeze z
                    per j < m-6: scalar G_j(1)' · scalar G_j(∞)' · squeeze ρ_j
                    scalar final_a' · scalar final_b'
lincheck            label flock-lincheck-v0 · squeeze α · squeeze β (const pin)
                    per j < k_log-6: scalar q_j(1)' · scalar q_j(∞)' · squeeze r_j
                    slice z_partial' (64) · squeeze z_skip'
opening             label flock-pcs-open-batch-v0                         (the algebraic phase's masking ends)
                    per claim: label flock-ring-switch-v0 · slice s_hat_v (128) · squeeze r'' (7)
                    squeeze γ (claims)
Ligerito level 0    label flock-ligerito-basis-f256-split-v0 · scalar target · bytes cap
  M1                label verity-flock-zk-l0-open-v0                      (level 0's masking begins)
                    per OOD sample: squeeze z (log_n) · scalar y' · bytes nonce · squeeze β
                    msg_0' (u_0 slice 2, u_2 slice 2)
                    per j < initial_k: squeeze r_j (F256) · msg_{j+1}' for j+1 < initial_k
  M1                slice e_0' (2) · slice e_1' (2) (rep r's pair) · squeeze ρ_0, ρ_1 (F256)
  M1                label verity-flock-zk-l0-close-v0                     (level 0's masking ends)
  M1                slice T' (2) · slice ybar (2 t_pad)
                    code-switch message · cap_1 · ... levels 1+, unchanged ...
M1 inner proof      label verity-flock-zk-inner-v0 · scalar final_c' · squeeze ε · slice [ρ, σ] · squeeze γ (10)
  M2                bytes τ's commitment (64) · squeeze β · slice Y (s+192) · squeeze queries (168)
proof bytes         bincode (Commitment, R1csProofLigerito with the masked fields, InnerProof)
~~~

Upstream never absorbs `final_c'`; M1 absorbs it first in the inner proof (§5). The claim order in the opening is ab, then
c, then the region claims (region-major, then the two link points).

## 2. The pads

Per rep, the pad count is

~~~text
s = 2·64 + 2(m−6) + 3 + 2(k_log−6) + 64 + (n_ood0 + 4·initial_k + 4) + 4
~~~

RoPE at `m = 25`: 298. `Layout` in `zk_veil.rs` is normative.

| indices | pad of | count (RoPE m=25) |
|---|---|---|
| `[0, 64)` | `round1_ab` | 64 |
| `[64, 128)` | `round1_c` | 64 |
| `128 + 2j + t` | multilinear round `j`, `t = 0: G(1)`, `t = 1: G(∞)` | 38 |
| `fa, fb, fc = 128 + 2(m−6) + {0,1,2}` | `final_a`, `final_b`, `final_c` | 3 |
| `fc + 1 + 2j + t` | lincheck round `j`, `t = 0: q(1)`, `t = 1: q(∞)` | 32 |
| `zp + i`, `zp = fc + 1 + 2(k_log−6)` | `z_partial[i]` | 64 |
| `lg_ood(i) = zp + 64 + i` | Ligerito level 0's OOD value `i` | 1 |
| `lg_msg(j, t) = zp + 64 + n_ood0 + 4j + t` | lane round `j`'s message, `t`: `u_0.c0, u_0.c1, u_2.c0, u_2.c1` | 24 |
| `lg_e(i, c)` (next four) | the rep's extra lane `i`'s sum (lane `64 + 2r + i`), coordinate `c` | 4 |
| `hab` (next) | `h_ab = h_fa · h_fb` | 1 |
| `tx, ty, tw = hab + {1,2,3}` | the sacrificed triple `x, y, w = x · y` | 3 |

The first `hab` pads mask sent values, and the last four are committed auxiliaries. The absorption order is
`Layout::absorbed()`: the rows above in order, minus `fc` (the inner proof's, §5). The algebraic phase is masked between
the zerocheck label and the opening's label. Level 0 is masked between `verity-flock-zk-l0-open-v0` and `-close-v0`.

**Randomness.** Everything comes from `ProverRng`, which is ChaCha20 keyed by the session's `ProverSeed` (a fresh
`getrandom` draw). `f128s(stream, index, n)` reads `16n` bytes from byte offset `32·index` of the stream. With
`base = rep << 32`:

~~~text
per rep r:
  h      = f128s(4, base, s)            then h[hab] := h[fa]·h[fb],  h[tw] := h[tx]·h[ty]
  mu     = f128s(5, base, s)            the dot-product mask
  pad_h  = f128s(4, base + 2^24, 192)   the pads codeword's padding
  pad_mu = f128s(5, base + 2^24, 192)
per session (both reps):
  mask slot, block b: f128s(1, b · 2^(mask_log−7), 2^(mask_log−7))    (M0's hook, uniform in --zk)
  Ligerito level 0 (one codeword, both reps open it):
    extra lane i < 4:     f128s(6, i << 32, L)            rep r folds lanes 2r, 2r+1 only
    lane l's padding:     f128s(7, l << 32, t_pad)        l < 64 + 4,  t_pad = 2·queries_0 (554 at m = 25)
~~~

The level-0 padding of the 64 witness lanes is shared by the reps, as is the codeword; each pair of extra lanes and its
padding belongs to one rep (§7).

Streams 1–3 belong to M0: mask, padding, and the injected-seed salt tape. None of this is derived from the transcript.

**Several tables** (`--session-tables J`, the CPU prover). Table `j` reads every stream above at ChaCha20 stream
`purpose | j << 32` (`ProverRng::for_table`). Its salt trees are ids `j << 32` (its level-0 tree) and up, below the next
table's (`LeafSalts::table`, the session's one salt key). So no two tables share a pad, a mask word, a padding coefficient or
a salt. Table 0's addresses are a one-table session's, and a one-table session's bytes are unchanged.

## 3. The pads commitment

The code is Reed–Solomon over GF(2^128). The message is `(h ‖ pad_h)` of length `L_h = s + 192`, read as coefficients
of `Σ m_i X^i`. The code is evaluated at the `N = 8 · next_pow2(L_h)` points `x_j = F128(lo = j + 1, hi = 0)`, which
gives RoPE `N = 4096`. There are two rows, `C(h ‖ pad_h)` and `C(mu ‖ pad_mu)`. Leaf `j` is the `hm96-sha512/v1` leaf of
`x = SHA-512(bytes(C_h[j]) ‖ bytes(C_mu[j]))`. Its salt is leaf `j` of a fresh tree id from the proof's `LeafSalts`
(ChaCha20 blocks `3j..3j+2` under the salt key, nonce `id`). The id comes from `next_id()`, so it is never 0 and never
reused. Nodes are `SHA-512(left ‖ right)`. The 64-byte root is the item after `verity-flock-zk-pads-v0`.

## 4. The constraints

The verifier re-runs upstream's zerocheck and lincheck verifiers on the masked proof (`zk_veil::replay`), drawing the
same coins and absorbing the same items. Every message-dependent quantity is an affine form `c + Σ a_i h_i` (`Lin`; over
F256, two of them, `Lin2`). Each check `lhs = rhs` becomes `lhs + rhs = 0`. The Lagrange weights come from upstream's
`interpolate_at_z_on_lambda` and `interpolate_at_z_combined`, evaluated on unit vectors.

| name | constraint (all `= 0`) |
|---|---|
| zerocheck c_eval | `Σ_i W_c(z)_i (round1_c'_i + h_{r1c(i)}) + final_c' + h_fc` |
| zerocheck final a·b | `G_final + final_a'·final_b' + final_a'·h_fb + final_b'·h_fa + h_ab` (`G_final` the running claim) |
| lincheck final | `running + Σ_i comb_i (z_partial'_i + h_{zp(i)})` (`comb` = `fold_alpha_batched` folded at the rounds' coins) |
| ring switch ab | `claim_check(w_ab, s_hat_v_ab) + Σ_i λ_i (z_partial'_i + h_{zp(i)})` (`λ` = φ8 weights at `z_skip'`) |
| ring switch c | `claim_check(w_c, s_hat_v_c) + final_c' + h_fc` |
| ligerito lane phase (c0, c1) | `claim_k + ρ_0 e_0 + ρ_1 e_1 + T'`, where `claim_0 = target + Σ_i β_i (y'_i + h)` and `claim_{j+1} = u_0 + r_j (claim_j + u_2) + r_j² u_2` over the masked lane messages (upstream's `RoundQuad256`) |
| triple ρ | `ε h_fa + h_tx + ρ` |
| triple σ | `h_fb + h_ty + σ` |
| triple sacrifice | `ε h_ab + h_tw + σ h_tx + ρ h_ty + ρσ` |

The ring-switch rows replace upstream's claim check on the ab and c claims: the verifier calls upstream's opening with
`claims[0..2] = claim_check(...)`, and every other part of the opening runs unchanged. The lane-phase rows replace the
Ligerito verifier's lane-round arithmetic, which starts again from `T'`. The triple rows bind `h_ab = h_fa h_fb`: with
`h_ab = h_fa h_fb + δ1` and `h_tw = h_tx h_ty + δ2`, the sacrifice row evaluates to `ε δ1 + δ2`.

## 5. The inner proof

It proves `⟨c, h⟩ = t` for the batched constraint, where `(c, t)` comes from `batch(cons, γ)` over the 10 rows:

1. P sends `final_c'`, a constant of the c_eval and ring switch c rows: sent after `γ`, it could be solved to cancel any
   failing row. `ε` ← V. P sends `ρ = ε h_fa + h_tx` and `σ = h_fb + h_ty`.
2. `γ` (10) ← V. Both sides form `(c, t)`. P checks `⟨c, h⟩ = t` on its pads and stops otherwise, before it sends anything
   more. P computes `τ = ⟨c, mu⟩` and sends its commitment
   `hm96::leaf(SHA-512("verity/flock-zk/tau/v1\0" ‖ τ), salt)` with a fresh 192-byte salt (`tau_commitment`, M2).
3. `β` ← V. P refuses `β = 0`. P sends `Y = (h + β mu ‖ pad_h + β pad_mu)`.
4. 168 query words ← V, reduced to `j = lo mod N`. P opens `(C_h[j], C_mu[j])` with salts and paths. The proof carries
   `τ` and its salt.
5. V checks that `τ` opens its commitment, `⟨c, Y[..s]⟩ = t + β τ`, `C(Y)_j = C_h[j] + β C_mu[j]` at every query, and each
   path to the pads root.

`τ` in the clear before `β` (M1's first version) would let a verifier test `⟨c, Y⟩ = t + βτ` as soon as `Y` arrives,
before the last coin; §9's extraction run needs every message before the last coin masked, committed or public.

**The prover checks before it speaks** (step 2, `zk_veil::SelfCheck`). On a failing constraint, `⟨c, Y[..s]⟩ + t + βτ`
is `⟨c, h⟩ + t`, a function of the witness, and the simulator covers only runs whose constraint holds. So the prover
stops with `PROVER-STOPPED: the batched constraint fails` before `τ`'s commitment, and the last thing it sent is `[ρ, σ]`.
An honest witness passes for every `γ`, so the honest prover's messages and proofs are unchanged. Two kinds of run skip
the check: the selftest's negative controls whose witness or pads fail by design (`plan_of` names them), whose proofs
the verifier must refuse, and the simulator's dummy-witness runs (§9), which must reach the proofs as a real run does. No
flag or variable turns it off.

**Soundness terms, for review.** Batching costs `1/|F|`, the sacrifice `1/|F|`, and `β` `1/|F|`. RS proximity is at
rate `L_h/N ≈ 0.12` with 168 queries: `(1 − δ)^168` at the unique-decoding `δ ≈ 0.44` is about `2^-140`, plus BCIKS20's
`N/|F|`. The pads bind before the first exposed value, because the root is absorbed before `r_skip`, and `final_c'`
before the batching coins. The base protocol's soundness then carries over as in VEIL Lemma 4.3: `v' − h` is a prover
strategy for the unmasked protocol.

## 6. The mask slot hides the masked claims' `s_hat_v`

`s_hat_v^p[v] = Σ_x eq(p_w, x) z[128x + v]` over word indices `x`, where `p_w = x_full[1..]` of the claim point, taken
LSB first. The audit checks this indexing by recomputing every region claim's `s_hat_v` from public data. The mask slot's
words (`2^(mask_log−7)` per block, all blocks, uniform bits `u`) add `Σ_{x ∈ mask} eq(p_w, x) u_{v,x}`. For the claim
points `p_1..p_k` seen so far (ab and c of each rep, up to 4), that sum is uniform on `F^k`, independently for every `v`,
exactly when `(eq(p_1, x), .., eq(p_k, x))` over the mask words span `GF(2)^{128k}`. The prover checks this rank at the
opening's label, before any `s_hat_v` leaves (`mask_rank_ok`, 2–4 ms), and stops otherwise. RoPE: rank 512 needed, from
1024 words. Region claims' `s_hat_v` are functions of public region data (`Stmt::region_s_hat_v`).

## 7. Ligerito level 0 (VEIL Figs. 8–9), and zero knowledge for the session

Without this, Ligerito sends witness functionals in the clear. At `m = 25` the audit finds per session:

- the level-0 rows, 446 KB;
- the recursive rows, 118 KB;
- the sumcheck messages, the OOD values and the final vector, 4.3 KB. One level-0 OOD value is `P̂(z)`.

What is changed:

1. **Per-lane padding** (`zk::lane_codeword`). Lane `l`'s codeword encodes `P_l + (c·pad_l ‖ 0) ‖ pad_l ‖ 0`: `t_pad`
   uniform coefficients at `[L, L + t_pad)` and `c` times them at `[0, t_pad)`, with `c = x` (`PAD_C`). That is, the
   padding polynomial is `(X_L + c)·μ` with `μ = Σ pad_i X_i` of degree `< t_pad`.
   - *The binary-field point:* in the additive NTT's novel basis `X_L = Ŵ_n`, which is 0 on the first half of the
     codeword's positions and 1 on the other half, which I checked on upstream's NTT. So VEIL's high-coefficient
     padding alone would leave half the opened positions unmasked.
   - `X_L + c` vanishes nowhere, and `X_0..X_{t_pad−1}` span the polynomials of degree `< t_pad`, so any `t_pad` distinct
     positions of a lane are uniform.
   - The committed word still determines `P_l` (its first `L` coefficients minus `c` times the next `t_pad`).
2. **Four uniform lanes** (`zk::EXTRA_LANES`), padded the same way, **a pair per repetition**: rep `r` owns lanes
   `64 + 2r, 64 + 2r + 1` (`zk::rep_lanes`), `R^r = (R_{2r}, R_{2r+1})`. The leaf holds 68 F128 values
   (`commit::commit_zk`: one tree over the same positions, so the same hm96 level-0 salts and C1's shared root).
3. **After the lane rounds** of rep `r` (challenges `r̄^r`), in this order:
   - P sends `e^r_i = ⟨w^r, R^r_i⟩` (exposed, padded), where `w^r = b(r̄^r, ·) + Σ_j β_j·eq(z_j, (r̄^r, ·))` is the folded
     next-stage weight, level 0's OOD claims included: the weight the running claim carries after the lane rounds (the
     sumcheck's `combined_basis`). They precede the coins that combine them, as in VEIL step 1;
     otherwise a prover could choose `e` to fit any claim.
   - V draws `ρ^r_0, ρ^r_1 ∈ F256`. P refuses them unless they are independent over F128, so `W^r : (a, b) ↦ ρ^r_0 a + ρ^r_1 b`
     is invertible on F128².
   - The fold becomes `y_1'^r = y_1^r + W^r R^r`, uniform on F256^L.
   - P sends `T'^r = ⟨w^r, y_1'^r⟩ = s + ρ^r_0 e^r_0 + ρ^r_1 e^r_1` (`s` the running claim) and
     `ybar^r = Σ_l eq(r̄^r, l) pad_l + W^r pad_{R^r}` (`t_pad` F256 values, `zk::ybar`).
   - The code switch and level 1 commit `y_1'^r`. The verifier's running claim restarts at `T'^r`.
4. **The level-0 row check** at rep `r`'s queries (the verifier is told `r`: `zk::Level0Params`):

   ~~~text
   ⟨row_q[..64], eq(r̄^r)⟩ + ρ^r_0 row_q[64 + 2r] + ρ^r_1 row_q[64 + 2r + 1] + Enc(c·ybar^r ‖ ybar^r)(q) = Enc(y_1'^r)(q)
   ~~~

   It is folded into the induced basis's enforced sum (`zk::enforced_extra`). The induced basis itself is unchanged. The
   identity holds at every position, as an identity of codewords.
5. **Exposed** (padded, §2): the level-0 OOD value, the six lane-round messages and `e^r`. The prover's lane chain becomes
   the two lane-phase rows (§4).
6. **Level 0's query count and padding.** The padded code is `RS[2L, L + t_pad]`, of rate `ρ' = (L + t_pad)/2L`, and the
   verifier's query bound is at that rate: `q_0 log₂(1/(√ρ' + 1/50))` bits per run (`zk_veil::level0_bits`, the soundness
   lane's accounting). `t_pad = 2 q_0` covers every distinct level-0 position the session opens; nothing else evaluates the
   padded lanes (the OOD value, the lane messages and `e` are masked by the algebraic pads, `ybar^r` by rep `r`'s own pair).

   | m | L | q_0 | t_pad | bits per run | the least q_0 |
   |---|---|---|---|---|---|
   | 25 | 2^12 | 277 | 554 | 102.7 | 268 |
   | 26 | 2^13 | 242 | 484 | 101.5 | 239 |
   | 27 | 2^14 | 229 | 458 | 100.9 | 227 |
   | 28–31 | 2^17–2^18 | 219 | 438 | ≥ 100 | 219 |
   | ≥ 32 | ≥ 2^19 | 218 | 436 | ≥ 100 | 218 |

   At `m = 25..27` the counts are the ones the soundness lane and the red team agreed (computed with 64 more padding
   symbols); elsewhere the least count. `--zk` refuses `m ≤ 24`, where no count reaches 100 bits at rate 1/2, and any
   count below 100 bits (`zk_queries0`).

**Zero knowledge for the session.** Fix every coin of the session (both reps' streams and the link points). The view is:
the pads roots, caps and paths (hm96 commitments); per rep, the masked algebraic values, `s_hat_v` of the masked and the
region claims, the level-0 OOD value, lane messages and `e^r` (all padded), `T'^r`, `ybar^r`, the rows at rep `r`'s
level-0 queries, levels 1+, and the inner proof. The prover's randomness is spent as follows, each part once:

| view | uniform through | scope |
|---|---|---|
| masked algebraic and level-0 exposed values | the pads `h` (a bijection, §2) | per rep |
| `s_hat_v` of the 4 masked claims | the mask slot, jointly over all 4 claim points (the rank check, §6) | session |
| region `s_hat_v` | public data | |
| the 64 witness lanes' entries at every opened level-0 position of either rep | their padding `μ_l`, `t_pad ≥` those positions | session |
| `y_1'^r`, hence every level-1+ message and row of rep `r` | `R^r` through `W^r` | per rep |
| `ybar^r` | `pad_{R^r}` through `W^r` | per rep |
| rep `r`'s pair entries at any opened position | none: fixed by the row-check identity from the three rows above | |
| the inner proof (`ρ, σ, τ, Y`, ≤ 168 of 192 padded columns) | its triple, `μ`, `pad_h`, `pad_μ` (VEIL Lemma 4.3) | per rep |
| unopened leaves, caps, paths | hm96 salts (fresh tree ids; level 0's one tree) | per tree |

The other pair's entries in rep `r`'s rows are fixed the same way, by the other rep's identity. So for every witness the
session's view is uniform subject to the public identities, which is how the simulator samples it (§8). Nothing is
spent twice. A pair shared by the two reps would be: `(y_1'^0, ybar^0)` and `(y_1'^1, ybar^1)` would both draw on one
`(R, pad_R)`, and `N·(·)^0 + (·)^1` with `N = W^1 (W^0)^-1` eliminates it. That was red-team-m1-zk's finding, a witness
functional at every level-0 position either rep opens and at the level-1 positions both open. With `k` extra lanes
shared by both reps, joint uniformity needs the `4 × k` matrix `[W^0; W^1]` to have rank 4, so a pair per rep (4 lanes)
is the minimum.

**Checked:**
- `zk_veil` test `exact::a_pair_per_rep_hides_the_witness_and_a_shared_pair_does_not`: at toy size (4 witness lanes of 16
  words, 3 queries per rep, 5 distinct positions), with the protocol's own `lane_codeword`, `pad_codeword`, `ybar` and
  `enforced_extra`. For fixed coins the view is F128-linear in the witness and the randomness, and it hides the witness
  exactly when the witness's images lie in the span of the randomness's. Even with all of both `y_1'^r` revealed, the
  per-rep pairs hide it; a shared pair does not, nor does `t_pad` below the distinct opened positions. The verifier's row
  check holds on the model.
- selftest `zk_session_relations_vary_with_the_seed` (RoPE `m = 25`, one coin seed): the finding's two functionals
  (`session_relations`: 521 F256 values at level 0, 42 at level 1) differ in every word between two prover seeds. A
  prover that folds rep 0's pair in both reps (rejected by the verifier) reproduces the leak: every word is equal.
- The statistics (§8) test both functionals as classes of one value per session.

**Assumptions.** The table's argument is exact (perfect hiding) for uniform prover randomness, given the commitments hide.
- **hm96-sha512 hiding is statistical, in the standard model** (`verity/primitives/commitments/hm96/PROTOCOL.md` §3: a
  leftover-hash-lemma argument over the Hankel family, which uses no property of SHA-512). Caps, paths and roots are
  functions of the leaves, so they inherit the bound. It needs:
  - uniform, independent salts, never reused: leaf `j` of tree `id` takes ChaCha20 blocks `3j … 3j+2` under nonce `id`, and
    `next_id()` is never 0 and never repeats (level 0 is the one tree both reps share, each leaf salted once); the pads
    trees and τ's commitment have their own salts;
  - the pinned key outside the `2^-64` fraction of bad keys: the common-reference-string step under `hash-derived-key`.
    The distance is then about `N·2^-192`, with `N` every leaf ever committed under the key (an `m = 25` session commits
    about 2^14.4), so a lifetime of `2^64` leaves stays within `2^-128`.
- **So M1 is statistical special honest-verifier zero knowledge for one session**, in the common-reference-string model,
  with distance about `N·2^-192`. The claim ids are `statistical-hiding`, `common-reference-string` and `hash-derived-key`.
  No random oracle is involved.
- **The one caveat is the randomness source.** Every pad, mask, padding coefficient, extra lane and leaf salt is ChaCha20
  output under the session's 32-byte seed from the OS (`ProverRng`, `LeafSalts`). The claim is statistical when that output
  is treated as uniform, as hm96 §3 treats such streams (and `os.urandom`); otherwise it is computational, under ChaCha20
  as a PRG.
- **Collision resistance** (`cr/sha-512`) enters only soundness (binding of the prover's trees) and M2's binding of the
  verifier's coin commitment (§9), not M1's hiding.
- **Refused coins.** The prover's refusals are exactly the ones the argument uses: `β = 0` (the inner proof),
  `ρ^r_0, ρ^r_1` dependent over F128 (`W^r` invertible), mask-slot weights of rank `< 128·(claim points)` (§6), and a
  zerocheck outer coin `r_j = 1` (the replay's round `j` divides by zero there, and the honest pads miss the inner proof's
  constraint by a witness-dependent amount). Each stops the prover before it sends anything that depends on the coins it
  reads, and reads only opened coins; `r_j = 1` is checked as the coin opens, in `zk_veil::Masking`'s draw. Its
  completeness loss: under uniform coins the honest prover refuses with probability at most `(#outer coins)·2^-128` per
  rep, `m − 13` outer coins. §5's self-check refuses no coin: an honest witness passes it for every `γ`.
  The mask slot must have `blocks · 2^(mask_log−7) ≥ 128 · 4` words for the rank to be reachable. Both sides refuse a
  statement, or a table of a split session, with fewer than `128 · 4 + 64` words when it is built, before `Hello`
  (`zk_shape_ok`). RoPE and GEMM have 1024 at every split tried: m ≥ 25 at their `k_log` means at least 8 blocks.

Negatives, each rejected on both reps (RoPE CPU selftest):
- another extra lane folded than the committed one;
- `T'` forged;
- `ybar` forged;
- `e` forged with an honest `T'`. Ligerito accepts this one, and only the inner proof's lane-phase rows reject it.

## 8. The simulator, evidence, and what M1 still needs

**The simulator** (`simulate_full`, per rep, fixed coins; both reps from one dummy session, so the session's structure is
the dummy's with the dummy's randomness). Its input is the public statement, the verifier's coins for the rep's stream
(every squeeze, in order) and the two link points. It builds:

1. uniform exposed values and uniform masked-claim `s_hat_v`;
2. the region claims' `s_hat_v` from public data;
3. the opening as the honest prover makes it on a dummy (all-zero) witness with fresh masks, against the same coins, with
   level 0's exposed values drawn uniformly. This is Goldreich–Kahan's dummy-witness run, restricted to the part whose
   distribution the masks make witness-independent;
4. the inner proof simulated for the constraints the verifier derives from all of the above (`zk_veil::simulate_inner`):
   `ρ, σ, Y` uniform, `τ = (⟨c, Y⟩ + t)/β`, and the opened columns uniform subject to `C(Y)_j = C_h[j] + β C_mu[j]`.

The verifier (`verify_zk_with` on a coin tape) accepts every simulated proof. The simulator refuses the coins the prover
refuses.

**Statistics** (`zkstat --zk --n 64`, then `tests/zk_stats.py`). Real and simulated transcripts are compared per class:
exposed values, masked `s_hat_v`, the inner proof, level 0's exposed values, `T'` and `ybar`, level-0 rows, and levels 1+,
and per session the two cross-rep relations of §7 (`rel_level0`, `rel_level1`). Each class gets byte-histogram,
bit-balance and cross-run-XOR tests. Positions constant in every real and simulated transcript are public structure:
F256 messages after the code switch are base-valued (5 such positions in the levels-1+ sample). A level-0 position both
reps open holds one row, so its second copy is excluded (one row of the sample under these coins). The control (all
prover randomness zero) must be distinguished.

Results for RoPE at `m = 25`, 64 sessions (128 transcripts) per kind:
- real and simulated agree on all 13 classes; the smallest of the 65 p-values is 0.009;
- the control is distinguished on all 9 masked classes, both relations included (p ≈ 0, every position constant);
- all 128 simulated proofs are accepted.

Evidence: `lanes/flock-zk/evidence/`.

**The audit** (`zkaudit --zk`). Every byte the verifier keeps, and the proofs' opened data (with the opening's batching
nonces and the commitment's parameters), is attributed to a class. The inner proof's round values are `final_c'`, `[ρ, σ]`
and `Y`, in that order (selftest `zk_audit_attributes_the_inner_proof`). The audit checks that:
- masked values differ from the control's;
- region `s_hat_v` equal their public recomputation;
- every Ligerito value differs from the control's;
- nothing is unattributed or in the clear (`attributed: true`).

This is attribution, not hiding: that a mask is applied, not that it hides. Hiding is §7's argument, its exact check and
the statistics' relation classes.

**What M1 still needs:**
- **The red team's re-review** of §4–§7 after its first review (F1: the reps' shared extra lanes, fixed in §7; F2: level
  0's query count, §7 item 6; F5: the prover's own lincheck fold is an argument of `replay`, never read from a proof).
- **Cost.** Measured on a 4-core VM, per rep, from loopback sessions (median of three timed runs):

  | | M0 prove | M1 prove | upstream under M1 | proof per rep |
  |---|---|---|---|---|
  | RoPE `m = 25` (4 and 16 instances) | 0.24–0.27 s | 0.31–0.33 s (+21–32%) | 0.26–0.27 s | 560 → 865 KB |
  | RoPE `m = 27` (64 instances) | 0.36–0.37 s | 0.46 s (+24–27%) | 0.39–0.40 s | 618 → 843 KB |

  - The prover's own replay takes its lincheck's folded `comb` (`LincheckProof::comb_partial`, never serialized). The
    remaining fixed part is 0.026 s of replay and 0.02 s for the pads commitment.
  - Upstream's share (+7% at `m = 27`) is mostly level 0's padding codewords, one full NTT per lane (68 × 2L). One
    interleaved NTT of the padded input from layer 0 would cost about `68/64` of upstream's encoding.
  - The proof grows by the inner proof, mostly its 168 paths (130 KB per rep), which could share a cap, plus `ybar`
    (18 KB), the four extra lanes' entries, and level 0's 277 queries in place of 218.

## 9. M2: the verifier's coin commitment and the single-rewind simulator

The design for malicious-verifier zero knowledge for one batched session, in the soundness lane's model (ASSUMPTIONS.md
§2 and §6, DESIGN.md §11.1): the verifier commits to every coin before the prover's first message, then opens them round
by round, and a Goldreich–Kahan simulator rewinds once, to just after that commitment. It is not yet zero knowledge
against every committed coin sequence: the prover must also refuse a zerocheck outer coin `r_j = 1` (see Status).

**The coin commitment** (`src/coin_tree.rs`, scheme `coin-tree/hm96-sha512/v1`, `SessionConfig::coin_tree`).
- The schedule is pinned in `Hello` (`coins: {scheme, streams, rounds_log, block}`): streams `session`, `circuit/rep0`,
  `circuit/rep1`; `2^8` rounds per stream; blocks of 512 coins. A RoPE rep uses 100 rounds and at most 277 coins in one
  (level 0's queries); the session stream's round 0 is the two link points.
- At `Hello` the verifier draws every block and a 192-byte salt per (stream, round) and commits: per block the chain
  `T_512 = SHA-512(TAIL)`, `T_i = SHA-512(CHAIN ‖ c_i ‖ T_{i+1})`; the leaf's digest `x = SHA-512(ROUND ‖ u32be(stream) ‖
  u32be(round) ‖ T_0)` and the leaf `hm96::leaf(x, salt)`; each stream's leaves under a SHA-512 tree (a subtree), and the
  streams' roots under one more, padded with `SHA-512(PAD)`. The 64-byte root is `Hello`'s answer (4 F128 words), before
  the prover's first message (`Commit`).
- Every answer is the round's `n` coins, `T_n`, the salt and the path (`Spec::opening_words(n)`), so an opening reveals only
  the coins the round uses. The prover checks every opening against the root before it uses a coin (`coin_tree::Checked`)
  and stops at one that does not open, before it answers anything that depends on it.
- This replaces M0's step 0 (a seed commitment with derived streams): no PRF. The prover's refusals (`β = 0`, the mask
  rank, dependent `ρ`, an outer coin `r_j = 1`) stay functions of committed coins.
- **Two properties, two roles.** *Hiding* of the unopened coins is what soundness uses: statistical in the standard model
  (hm96 `PROTOCOL.md` §3), since every block and salt comes from `/dev/urandom`; no random oracle. *Binding* is the one
  assumption M2's zero knowledge adds: SHA-512 collision resistance (`cr/sha-512`), which holds whatever salts V* chooses
  (hm96 §4 holds for every key), so a rewound V* cannot open another coin. Against quantum verifiers the commitment must
  be collapse-binding (ASSUMPTIONS.md §6).

**The simulator** (`gk_simulate` in `flock-circuit.rs`), for every verifier V* as a black box on its random tape
(rewinding V* is a fresh instance on the same tape, replayed to the same point):
1. **Extraction.** The honest prover on a dummy witness against V*: every private word zero, the public region words
   written in (`Stmt::write_regions`: every row's `b ‖ c`, the output rows' included), the mask slot uniform, fresh masks. Every message
   before the proofs is masked, committed or public (§7, §5), and the region claims' `s_hat_v` are the public ones, so this
   run is distributed as a real one until the proofs, which follow the last coin. If it stops before the proofs (V*
   aborted or cheated, or the prover refused a degenerate coin), its view is the output.
2. **The extracted coins.** Every opening V* sent, checked against its commitment; the link points and each rep's rounds.
3. **Estimation** (`zk_veil::gk_estimate`). Fresh dummy runs against V*, with no cap and not counting the extraction run,
   until `t` of them reach the proofs; `p̃ = t / runs`. It runs only after an extraction run that reached the proofs
   (probability `p`), so it costs `t` runs in expectation.
4. **Rewinds**, at most `4t/p̃ = 4·runs`, each to just after the coin commitment and each with a **fresh dummy opening**
   for the extracted coins (`fresh_dummy`: the honest prover on the dummy witness with fresh masks, level-0 padding, extra
   lanes and salts, against a local coin server that commits to those coins; no interaction with V*). So the rewinds are
   independent trials, each succeeding with V*'s completion probability `p`. Each rewind is V* from the start of the
   session, answered by a scripted prover whose messages are the per-rep SHVZK simulations (`simulate_full` on that fresh
   opening, §8, each rep with its own pair). A `RecordingChallenger` recovers each round's message by replaying the
   verifier on the simulated proof. Every coin V* answers is checked against its commitment. The first rewind that
   reaches the proofs is the output. A rewound V* that opens another coin than it did in step 2 is a binding failure, and
   the simulator fails; so does exhausting the bound.

**The parameter `t` grows with the security parameter:** `t = Θ(λ)`, so that `p̃ ∈ [p/2, 2p]` except with probability
`2^-Ω(λ)`, and then all `4t/p̃ ≥ 2t/p` rewinds fail with probability at most `e^-2t`. A fixed `t` would leave a failure
probability that is noticeable against some V* (red-team-m2-zk G2, G3). The tests run at `t = 4` (`zkrewind --gk-t`); that
is a test's λ, not a bound for any production claim.

Against the theorem's statement (ASSUMPTIONS.md §6): one extraction run of the honest prover on a dummy witness (step 1,
with the estimation's fresh repeats of it); rewinding only to just after the coin commitment (step 4, the one rewind point,
repeated under Goldreich–Kahan's abort handling with independent trials); expected polynomial time (`O(t)` runs in
expectation), with failure probability `2^-Ω(t)` for `t = Θ(λ)`; resting on special HVZK for
every fixed coin sequence (§7's session argument, with the extracted coins); the tables' simulations composing through
independent masks (per rep: the pads and the pair; per session: the mask slot and the witness lanes' padding, which §7
spends once).

**Verifier strategies** (`VStar`): honest (its coins from its tape); `abort` at the first round whose message, hashed with
the tape, falls below a threshold (about half the sessions); `adaptive-coins`, which at such a round answers a coin it
picks from the prover's message under the committed coin's opening (the prover stops there); `zero-beta`, which commits
to `β = 0` at rep 0's β round (the prover refuses it before sending `Y`); and two gated ones, which continue for sure if
one message hashes "good" (half the time) and otherwise with probability `ε = 1/16`: `cap-gated` on the witness cap
(red-team-m2-zk's), `late-gated` on rep 0's round with `T'` and `ybar`. A simulator that reused messages across rewinds
fails against the gated ones. `zkrewind --zk --n N` prints N real and N simulated views per strategy, and
`tests/zk_rewind_stats.py` compares how they end (a permutation test), their verdicts, rep 0's first 256 masked values,
every received proof's checked fields and the session relations of §7.

**The rewinds are independent trials** (`zkgk --zk --attempts A --reuse-dummy`): for each gated V*, conditioned on an
extraction run whose gate was bad, A rewinds with fresh dummy openings complete at the rate of A real sessions (a Fisher
test in `zk_rewind_stats.py`), while A rewinds on the extraction run's own dummy opening (red-team-m2-zk G1, the defect)
complete at about ε, which shows the test's power.

Results for RoPE at `m = 25`, 40 real and 40 simulated views per strategy, one V* tape, `t = 4`, `ε = 1/16`:

| V* | complete views, real / simulated | how views end (p) | extraction runs reaching the proofs | estimation runs | rewinds | failures |
|---|---|---|---|---|---|---|
| honest | 40 / 40, all accepted | identical | 40 of 40 | 160 | 40 | 0 |
| abort | 26 / 22, all accepted | 0.18 | 22 of 40 | 157 | 47 | 0 |
| adaptive-coins | 23 / 28, all accepted | 0.13 | 28 of 40 | 175 | 52 | 0 |
| zero-beta | 0 / 0 (all stop at the β round) | identical | 0 of 40 | 0 | 0 | 0 |
| cap-gated | 22 / 23, all accepted | 1.0 | 23 of 40 | 193 | 55 | 0 |
| late-gated | 24 / 23, all accepted | 1.0 | 23 of 40 | 176 | 45 | 0 |

Rep 0's online values, the proofs' checked fields (11 classes) and the session relations are indistinguishable real
against simulated wherever there are enough views, with no value reused across reps and a column the inner proof opens
twice counted once (smallest of 363 p-values 0.003, α = 10^-3 per test; as one family, Bonferroni-adjusted p = 1). The rewinds are independent trials (`zkgk`, 40 attempts):

| gated V* | real sessions complete | fresh rewinds complete (p) | rewinds on one reused dummy (p), the defect |
|---|---|---|---|
| cap-gated | 22 | 24 (0.82) | 5 (1.1·10^-4) |
| late-gated | 25 | 25 (1.0) | 40 (1.2·10^-5) |

The reused opening is caught in both directions: cap-gated's rewinds then complete at about ε, and late-gated's replay the
extraction run's lucky pass every time. Evidence is kept in the private store.

Selftest negatives: `zk_verifier_coin_off_its_commitment` (the prover stops at rep 0's round 3),
`zk_verifier_commits_beta_zero` (it stops at the β round, 98), `zk_verifier_commits_outer_coin_one` (it stops at the
round that opens the outer coins), `zk_dummy_witness_passes_the_region_claims` (the dummy is
rejected by the inner proof alone), `zk_failing_witness_stops_before_tau` (an honest plan's prover on a witness with a
flipped unit bit stops after the batching coins: rep 0's stream ends with `[ρ, σ]`, and no `τ` commitment, `β` or proof
goes out).

**Still to do.**
- **C1's level-0 root check** stays: both reps prove one witness, level 0's randomness is drawn once per session and
  belongs to its one tree, and the pads trees take `next_id() ≥ 1`.
- **GPU.** `--zk` is CPU-only. The mask slot fill, §7's padding and extra lanes, and `commit_zk` belong on the device,
  expanded from ChaCha20 like the hm96 salts.
- **Several tables per session** (the public-circuit ZK proof's §2.3 and gap 1) are on the CPU prover, `--session-tables J`:
  - the draw's units split into J equal tables `circuit-00`, `circuit-01`, ... (`session_tables`), under Σ over every
    table's Σ. The server refuses at configuration a Bernoulli law (its count is random), a count J does not divide, and
    tables `--zk` cannot prove. The session's law is the record's `unit_draw`: each table's header carries only its part
    as a subset draw;
  - randomness per table (§2) and a rank check per table (§6). Per table is the joint check here, because no claim or
    message reads two tables' witnesses or masks. The tables' witnesses may be correlated: two drawn units in different
    tables can read the same row;
  - one `Commit` carrying every table's level-0 root: a later table commits its root before any stream opens, and its reps
    stop unless they bind it;
  - a coin subtree per table and rep;
  - every proof after the session's last coin (the barrier).

  The selftests are `tables_two_in_one_session`, `tables_proofs_before_the_last_coin_detected` (a prover without the
  barrier is accepted by the verifier and caught by the order check), `zk_tables_share_no_randomness` and
  `tables_unsplittable_draw_refused`. The device prover still proves one table per session.

  **The simulator over several tables.** `zkrewind` and `zkgk` take `--session-tables J`: `gk_simulate` extracts,
  simulates and replays every table's streams in the prover's order. `zk_rewind_stats.py` tests each table's proofs and
  relations as their own classes, and fails any value that table j shares with table 0 in every view (`across_tables`).
  At J = 2 on RoPE (two tables of m = 25), 40 real and 40 simulated views per strategy:

| V* | complete views, real / simulated | how views end (p) | extraction runs reaching the proofs | estimation runs | rewinds | failures |
|---|---|---|---|---|---|---|
| honest | 40 / 40, all accepted | 1.0 | 40 of 40 | 160 | 40 | 0 |
| abort | 14 / 18, all accepted | 0.70 | 18 of 40 | 181 | 64 | 0 |
| adaptive-coins | 17 / 14, all accepted | 0.74 | 14 of 40 | 147 | 51 | 0 |
| zero-beta | 0 / 0 (all stop at table 0's β round) | identical | 0 of 40 | 0 | 0 | 0 |
| cap-gated | 20 / 24, all accepted | 0.51 | 24 of 40 | 194 | 49 | 0 |
| late-gated | 21 / 22, all accepted | 1.0 | 22 of 40 | 164 | 35 | 0 |

  - No value is shared across the tables in any strategy.
  - As one family of 688 tests the run passes: the smallest p-value is 3.2·10^-4, Bonferroni-adjusted 0.22.
  - That p-value, the one test below α = 10^-3, is the real side's xor test on table 1's `inner_y` in the abort strategy,
    which reads the real views alone. Rerun with 40 fresh views per kind, it is 0.30, and every one of the abort
    strategy's 136 tests passes.
  - The rewinds are independent trials (`zkgk`, 40 attempts). cap-gated: 22 real sessions complete and 21 fresh rewinds
    (p = 1.0), against 1 rewind on a reused dummy. late-gated: 16 real and 24 fresh (p = 0.12), against 40 reused.
- **Review.** A red team on §5, §7 and this section, then the Lean reference prover and its differential test, before any
  `COMPLETE_*` class.

## 10. The firewall's send check: nothing leaves the GPU that the firewall did not fix

In a recursive session the developer runs a **firewall** it trusts between its workers, which nobody trusts, and the
auditor (the coin server and the verifier). The workers may run any kernel; the question is what they can make leave. A
**free choice** is any bit a worker can vary that reaches the auditor, or the proof bytes, without the firewall having either
computed it itself or held it to a relation that leaves the worker nothing to choose. The send check is the rule, per
message kind, by which the firewall removes every free choice.

Where the boundary runs:
- **Inner phase** (`rec/inner-transcript/v0`, ZK off): the whole M0 prover is the worker, and `verity_flock.rec_live`'s
  proxy is the firewall. The coin server receives only the public schedule and salted hashes (§10.3).
- **Outer phase** (`flock-circuit prove --zk --gpu`): the device (`flock_cuda_prove_circuit`) is the worker; the host side
  of `flock-circuit` (the masking challenger, the live stream, `zk_finish`, and the send check between them) is the firewall. The
  device reaches the firewall through three channels: each round's frames (`rounds(content, n)`, at its squeeze), the
  level-0 root (`hook(root)`, at the session's `Commit`), and the returned `(R1csProofLigerito, Commitment)`. The firewall
  checks all three. The device holds the witness, the level-0 hiding (`Level0Zk`: padding and extra lanes) and the whole
  `LeafSalts` key (32 bytes, from which every tree's salts expand: level 0's, levels 1+'s and the pads tree's), but never the
  pads `h`, `mu` or the inner proof's randomness. Holding the key fixes nothing it sends: every salted leaf and path that
  leaves is the recheck's or a recomputed tree's (§10.1), and the pads tree is the firewall's own.

### 10.1 The rule, outer phase (one rep, in §1's order)

Five kinds of answer: **constant** (the firewall holds the only allowed value), **recomputed** (the firewall computes the value
from the host witness, its own randomness and the coins, and refuses any other), **masked** (the firewall adds a pad the device
never sees), **the firewall's own** (the device never touches it), and **relation** (a check that fixes nothing by itself,
kept as an early tripwire).

| item (§1) | rule | why the device has no free choice |
|---|---|---|
| the frame skeleton: domain, every label, each item's op, kind and length, every round's coin count | constant | a function of the statement's shape; the firewall refuses any other label or item |
| statement digest (`bind_statement`) | constant | the digest of the statement the firewall loaded |
| witness cap: `Commit`'s root, `bind_statement`'s cap, Ligerito level 0's `bytes cap` | recomputed | the level-0 recheck (§10.2): `commit_zk` of the host witness under the session's level-0 salts and hiding |
| pads label and root | the firewall's own | the masking challenger inserts them |
| masked algebraic values: `round1_ab'`, `round1_c'`, `G_j`, `final_a'`, `final_b'`, `q_j`, `z_partial'` | masked | the wire carries `v + h` with `h` uniform and host-only, so it is uniform whatever `v` is. `v` enters the view only through §4's rows, which the self-check holds before anything that depends on them leaves (§5); a `v` that breaks a row stops the session |
| `final_c'` | the firewall's own | the inner proof sends it, from the host |
| `s_hat_v` of the masked claims (ab, c) | recomputed | `Σ_x eq(p_w, x) z[128x + v]` over the host witness (§6); the claim points are coins |
| `s_hat_v` of the region claims | recomputed | `Stmt::region_s_hat_v`, from public data |
| Ligerito's `target` | recomputed | a function of the `s_hat_v` and the opening's coins (`r''`, `γ`) |
| level 0's OOD value `y'`, lane messages `msg_j'`, `e^r'` | masked | as the algebraic values (`lg_ood`, `lg_msg`, `lg_e`) |
| every grinding nonce | constant `0` | live coins need no grinding (the device's live challenger absorbs nonce 0) |
| `T'^r` | recomputed | `T'^r = ⟨w^r, y_1'^r⟩ = ⟨w^r, y_1^r⟩ + ρ^r_0 e^r_0 + ρ^r_1 e^r_1` with the honest `e^r = ⟨w^r, R^r⟩`: `w^r` from the coins and claims, `y_1^r` the fold of the host witness's lanes, `R^r` the host's extra lanes. The lane-phase relation alone (`claim_k + ρe + T'`, on the device's unmasked values) does not fix it: a device that shifts `e` by `δ` and `T'` by `ρδ` keeps the relation and sends 256 bits of its choice |
| `ybar^r` | recomputed | `zk::ybar` from the session's level-0 padding (host), `w^r` and `ρ^r` |
| levels 1+: the code-switch message, every cap, OOD value, sumcheck message and nonce | recomputed | functions of `y_1'^r`, the coins and the salts the host expands for trees `1..R` |
| the inner proof: `[ρ, σ]`, `τ`'s commitment, `Y`, its openings | the firewall's own | `zk_finish`, from the pads on the host, over §4's rows replayed from the passed values with the lincheck's comb the firewall folds itself: a device's comb would choose the rows the inner proof proves, so the firewall drops it (the outer proof never carries it; the verifier folds its own) |
| the proof bytes | recomputed | every transcript field equals what the firewall passed; level 0's rows and paths are the recheck tree's at the queries the coins fix; levels 1+'s are the recomputed trees' |

So the rule is: **the firewall masks the algebraic phase and level 0's lane rounds, and recomputes the polynomial
commitment's outputs** (the level-0 commitment, the opening's `s_hat_v` and `target`, and everything from `T'` on), and
folds the lincheck's comb for its own relation check, as the verifier does. It does not recompute the zerocheck or the
lincheck, the bulk of the device's arithmetic. What it recomputes is a CPU Ligerito: the commitment over the whole
witness, and the opening's basis fold `w^r`, which `T'`, `ybar` and levels 1+ all read.

**What a refusal costs.** A refusal stops the session, which the auditor sees, at a message the device can move: §10.5 bounds
what that reveals over the whole recursive session. A firewall that refuses is evidence against its rack. Timing (when a
message leaves) is not in this rule; the network warden covers it.

### 10.2 The level-0 recheck

Before the session's `Commit` the firewall recomputes the level-0 commitment itself: `commit_zk(z, params, level0)` over the
host witness (`FC_HOST_WITNESS=1`: the firewall assembles the witness on the host and uploads it whole, so the witness the
recheck commits is its own; the assembly is counted in the session's prove time, as the device's witness kernel is
without it), under `hm96::begin` with the session's level-0 salts (`LeafSalts::level0_id`), with the `Level0Zk` the host
drew. It refuses `FIREWALL-REFUSED: level-0 root` when the device's root differs, before anything leaves. It keeps the tree, so
the proof's level-0 rows and paths are checked against it at the end. Its cost is reported per statement as `FIREWALL` lines
(`t.firewall_recheck` wall seconds, `cpu.firewall_recheck` the process's CPU seconds over it), beside the device's
`t.encoding_commitment`.

### 10.3 The rule, inner phase (`rec_live`)

| item | rule | why the M0 prover has no free choice |
|---|---|---|
| `Hello` | constant | the firewall compares the prover's, Σ aside, with the pinned schedule's as canonical JSON (`rec_live.canonical`: sorted keys, no spaces, so `2` and `2.0` differ) and forwards the schedule's own, never the prover's text or object |
| `Commit`'s level-0 roots | salted | each 64-byte node of each root reaches the coin server as a registered row's `b ‖ c` (`rec_live.top_rows`: its 32 u16 words, then 32 zero words; `hm96-sha512/row/v1`) under its own fresh 192-byte salt; the roots and salts stay in `private.json` |
| a round's caps (its `BYTES` frames that commit a tree) | salted | as the roots, one fresh salt per node |
| a round's free message | salted | `c_k`, its rows committed under fresh salts (as before) |
| statement digest, labels, domain, nonces | dropped | public constants or functions of the hidden statement; never forwarded |
| `Link`'s `y` | constant `0` | the schedule fixes its length (`link.y` elements of 16 bytes); the firewall refuses any other byte length, and any nonzero byte |
| the order of `Hello`, `Commit`, `Link` and `Finish` | constant | the firewall itself refuses a `Commit` before `Hello` or after the first, a `Link` before `Commit` or after the first, and a `Finish` before the schedule's last round (counted from its own streams, never by reading the coin server), each before any call on the coin server: a request out of its place reaches the coin server as a plain stop at the same point |
| stream names, `Open`'s parent and position, each round's coin count and the counts of its caps' nodes and message rows | constant | the pinned schedule's; the firewall refuses any other. The coin server's record lists the streams by name, as the schedule does, never in the order the prover opened them. An `Open` may come any time before its stream's first round, so the firewall keeps it and opens the stream at the coin server just before that round: neither the coin server's calls nor its record show whether or when the `Open` came, and a stream is listed from its first round |
| how the session ends | `finish` or `stop` | every session served is written and counted toward `--sessions`, with `end` `finish` or `stop` (a refusal, or a session that ends without `Finish`: closed, reset or cut inside a frame), never why |
| a request the firewall cannot read, or the firewall's own fault | refusal | any exception while handling a request stops the session (`FIREWALL-REFUSED: a request the firewall cannot read (…)`), never the firewall |

The schedule is `rec_live --schedule FILE`, and the proxy refuses to serve without one. `rec_live --schedule-for STATEMENT`
derives it before any session from `flock-circuit statement`'s record of the public file alone (`schedule_for`: `dense_m`,
`k_log`, `nbl` and `regions`, the four numbers the schedule reveals anyway), and `serve` writes it to `schedule.json` before
the first session. No session pins it: a schedule taken from a session the prover ran would be the prover's choice
(`--schedule-of PUBLIC_JSON` prints a record's schedule, to compare with the one it was held to). The record says how its tops
are salted (`"tops": "hm96-sha512/v1/node-row"`, `rec_live.TOPS`). A refusal (`FIREWALL-REFUSED: …`) stops the session: every later
request is refused with the same reason, `Finish` answers not accepted, and only `public.json` (what the coin server had
received, `end: "stop"`) is written.

A salted node's opening (node and salt) is the prover's. V*'s inputs (`rec_vstar.root`, `caps`) read the clear tops from
`private.json` and refuse `FIREWALL-REFUSED: tree top` when one does not open its commit string in `public.json`, or when the
record's tops are not salted. The outer circuit does not yet climb to the committed tops: `rec_outer` still registers the
clear cap nodes. A node is committed as a registered row is, so climbing costs `RecOpen` no hashing of its own: the verifier
builds a registered value from the coin server's `b ‖ c` (one row a node, under the firewall's salts), and each opening's
`out` reads the row of the node it climbs to. The firewall pays one SHA-512 a node more than for a bare node (the row
digest).

### 10.4 The prototype (`FC_FIREWALL=1`)

`flock-circuit prove --zk` with `FC_FIREWALL=1` runs the firewall (`firewall.rs`, and `Firewall` in `flock-circuit.rs`) on the GPU path and
the CPU path alike, for a statement of one table; it refuses to start without `FC_HOST_WITNESS=1`. It is stricter than
§10.1 in one place: the shadow recomputes the whole opening, level 0's lane rounds included, in lock step with the device.
Each of the device's opening frames must equal the shadow's byte for byte (`firewall_opening_rounds` counts the rounds
compared), and the opening that leaves is the shadow's. The lincheck's folded comb is the firewall's (`zk_finish` folds it),
and `final_c'` is computed from the passed values. The algebraic phase's frames are held to the skeleton (domain, labels,
each item's op, kind and length, the coin counts). At the end the returned proof must have the statement's shape (each
zerocheck and lincheck vector's length), the replay of the values the firewall passed must give the proof's exposed values, and
the zerocheck and lincheck that leave are rebuilt from the passed values (`zk_veil::from_exposed`), as the opening that leaves
is the shadow's: nothing of the device's returned proof leaves.

`FC_FIREWALL_TAMPER` makes the device dishonest in one named way. Each is refused by name, before the tampered message leaves
or, for a masked value (which leaves as `v + h`, uniform whatever `v` is), before the inner proof does:

| `FC_FIREWALL_TAMPER` | the device | refusal |
|---|---|---|
| `root` | flips a bit of its level-0 root | `FIREWALL-REFUSED: level-0 root: the device committed … where the recheck has …` |
| `witness` | proves a witness one bit off the host's | `FIREWALL-REFUSED: level-0 root: …` |
| `label` | flips a label byte in the algebraic phase | `FIREWALL-REFUSED: algebraic phase, round 0, frame 1: label …` |
| `opening` | flips an observed value in the opening's first round | `FIREWALL-REFUSED: opening round 0, frame …: the device's observe … differs from the firewall's in its payload` |
| `algebra` | flips a masked value, in its frame and its proof alike | `PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)`: `zk_finish`'s self-check (§5) |
| `algebra-proof` | flips a masked value in its returned proof only | `FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` |
| `shape` | appends a value to its returned proof's `round1_ab`, past the ones the firewall compares | `FIREWALL-REFUSED: the proof's shape (…) is [65, 64, …] where the statement's is [64, 64, …]` |

The inner phase's controls are V*'s: `rec_inner.py forge --what top` (a clear top that does not open the salted one the coin
server received) and `--what unsalted` (a record of clear tops), refused `FIREWALL-REFUSED: tree top: …` by `rec_vstar`.
`85-rec-reprice.sh` runs both phases on node 1 (`STEP=icheck`; `STEP=oprove MODES=dev,host,firewall CONTROLS=L6`).

**What it costs** (node 1, one RTX PRO 6000, the prover's 16 cores shared with the firewall; K = 4096, V*'s eight `--zk`
statements L0–L6 and the algebra, m = 28–32; medians of 3 sessions after 1 warm, summed over the eight): prove 46.8 s
firewalled against 7.4 s with the host witness unfirewalled and 6.7 s with the device's witness; the proof bytes are the same. The
firewall's CPU is about 176 CPU seconds a session (the firewalled process's over the unfirewalled one's). Of the 39 s the firewall adds to
prove, the level-0 recheck is 5.2 s (51 CPU seconds; L0's alone 1.8 s and 17.8), the device's wait for the shadow opening
13.4 s, the firewall's own comb fold 10.5 s, and the rest the device's host phases slowed by the firewall's threads on the same
cores. All of it is CPU work. Three changes not yet made would shorten it: the firewall on cores of its own (the contention,
about 10 s), the recheck a session ahead (it reads only the host witness and the firewall's own salts and hiding), and the
device skipping the opening the shadow replaces (freeing the device's share of the cores). The shadow's own opening (15.7 s
over the eight, under contention) and the comb fold stay on the session's path.

### 10.5 What a stop reveals: the whole recursive session

A recursive session is the inner phase's session and the outer phase's sessions, one per V* statement (at K = 4096 eight: the
levels L0–L6 and the algebra). With §10.1 and §10.3, every message of a session that reaches the auditor is fixed, and so is
their order: `Hello`, `Commit`, `Link`, each stream's `Open` and rounds, the proofs, `Finish` (`Ping`, and `Register` in a
session that draws no units, change no record: the inner firewall answers `Register` itself). An inner `Open` may come any
time before its stream's first round, but the firewall passes it to the coin server only at that round (§10.3), so where it
came never reaches the auditor. A `Commit`, `Link` or `Finish` out of its place is refused by the firewall before any call
on the coin server (§10.3), so it reaches the auditor as a plain stop. What a worker still chooses is
whether and where a session stops: a refusal it provokes, or a session that ends without `Finish` (a crash, a hang, a cut
connection). The auditor sees every stop:
- inner phase: `rec_live` writes every session it serves, with `end` `finish` or `stop` (a refusal and a session that ends
  without `Finish` alike, never why), and counts it toward `--sessions` (§10.3);
- outer phase: `flock-circuit serve` writes every connection's record, finished or not, and counts it toward `--sessions`.
  The firewall sends `Finish` after a refusal and a crash or a cut does not, so an outer record also shows which of the two
  ended it.

**A stop ends its session**, never part of it: the inner firewall refuses every later request, and the outer firewall sends
nothing after its refusal but `Finish`. The outer phase starts only from a finished inner session (V*'s statements are staged
from its record: `rec_outer`, `rec_residuals`), so a stop in the inner phase ends the recursive session. A stop in the outer
phase ends its own session alone: nothing carries it to the other V* statements, which `85-rec-reprice.sh` proves each
against its own `serve`.

**The bound.** A session of `M_p` messages ends in one of `s_p + 2` ways: finished, never opened, or stopped after its `k`-th
message (`k < M_p`), with `s_p = M_p` stops in the inner phase and `s_p = 2 M_p` in the outer (with `Finish` or without).
Counted from serve's and the proxy's records of one recursive session at K = 4096 (2 reps a session): the inner session's
286 messages (139 rounds a rep) and the outer sessions' 268, 266, 266, 242, 242, 236, 232 and 288 (L0–L6, the algebra),
2326 in all.

| a stop | what it reveals, at most | at K = 4096 |
|---|---|---|
| ends the recursive session (no later session of it runs) | `log₂(1 + Σ_p (s_p + 1))` | 12.1 bits (11.18 = `log₂(2326 + 1)` without the outer's `Finish` bit and the never-opened sessions) |
| ends its own session only (each V* statement on its own, as today) | `Σ_p log₂(s_p + 2)` | 80.1 bits (72.1 = `Σ_p log₂(M_p + 1)` without them) |

Today the second row holds: a stop in the outer phase does not end the others. The first needs an auditor's rule, that a
recursive session ends at its first stop (no further V* statement of it is proved), which no code enforces yet.

**Retries.** Every stop is recorded and counted, in both phases, so the auditor knows how many times a recursive session was
run, and each run it allows reveals its own ending: `r` stopped runs before an accepted one reveal at most `r + 1` times the
row's bound. The auditor chooses `r`, since it decides whether a rack runs again after a stop; with `r = 0` a rack whose
firewall stops once has no second run.
