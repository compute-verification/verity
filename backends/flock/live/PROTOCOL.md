# verity/flock-circuit: the masked protocol (M1)

What `flock-circuit --zk` sends, where every mask comes from, and what the verifier checks. This is the layout the Lean
reference prover must reproduce byte for byte (soundness DESIGN.md §6: a deterministic function of statement, witness,
prover seed and verifier coins), and the statement its zero-knowledge theorem is about.

Code:
- `src/zk_veil.rs`: masking, replay, constraints, the inner proof, the simulator's pieces.
- `src/bin/flock-circuit.rs`: `prove_cpu_zk`, `zk_constraints`, `verify_zk_with`, `simulate_full`, `zkstat`, `zkaudit`.
- `../flock-zk-b684b12.patch`: Ligerito level 0's hiding in upstream Flock (`pcs::ligerito::zk`, `commit::commit_zk`, the F256
  split prover and verifier).

**Status.** CPU prototype, one template (RoPE d64), lane flock-zk. Every message a session sends is masked, committed or
public, and the audit (`zkaudit`) attributes every byte. The target notion is VEIL's special honest-verifier zero
knowledge for the whole session (both repetitions), meaning one simulator for every fixed coin sequence (DESIGN.md §11),
computational under the assumptions of §7. Malicious-verifier zero knowledge comes from M2's coin commitment and single
rewind (§9). The first red-team review (red-team-m1-zk) found that the two reps shared level 0's extra lanes; §7 now gives
each rep its own pair. The proof class stays `NON_ZK_PROOF` until the red team re-reviews and the Lean proof covers it.

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
M1 inner proof      label verity-flock-zk-inner-v0 · squeeze ε · slice [ρ, σ] · squeeze γ (10)
                    scalar τ · squeeze β · slice Y (s+192) · squeeze queries (168)
proof bytes         bincode (Commitment, R1csProofLigerito with the masked fields, InnerProof)
~~~

`final_c'` is sent only in the proof: upstream never absorbs it. The claim order in the opening is ab, then c, then the
region claims (region-major, then the two link points).

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
`Layout::absorbed()`: the rows above in order, minus `fc`. The algebraic phase is masked between the zerocheck label and
the opening's label. Level 0 is masked between `verity-flock-zk-l0-open-v0` and `-close-v0`.

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

1. `ε` ← V. P sends `ρ = ε h_fa + h_tx` and `σ = h_fb + h_ty`.
2. `γ` (10) ← V. Both sides form `(c, t)`. P sends `τ = ⟨c, mu⟩`.
3. `β` ← V. P refuses `β = 0`. P sends `Y = (h + β mu ‖ pad_h + β pad_mu)`.
4. 168 query words ← V, reduced to `j = lo mod N`. P opens `(C_h[j], C_mu[j])` with salts and paths.
5. V checks `⟨c, Y[..s]⟩ = t + β τ`, `C(Y)_j = C_h[j] + β C_mu[j]` at every query, and each path to the pads root.

**Soundness terms, for review.** Batching costs `1/|F|`, the sacrifice `1/|F|`, and `β` `1/|F|`. RS proximity is at
rate `L_h/N ≈ 0.12` with 168 queries: `(1 − δ)^168` at the unique-decoding `δ ≈ 0.44` is about `2^-140`, plus BCIKS20's
`N/|F|`. The pads bind before the first exposed value, because the root is absorbed before `r_skip`. The base protocol's
soundness then carries over as in VEIL Lemma 4.3: `v' − h` is a prover strategy for the unmasked protocol.

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
   - P sends `e^r_i = ⟨b(r̄^r, ·), R^r_i⟩` (exposed, padded). They precede the coins that combine them, as in VEIL step 1;
     otherwise a prover could choose `e` to fit any claim.
   - V draws `ρ^r_0, ρ^r_1 ∈ F256`. P refuses them unless they are independent over F128, so `W^r : (a, b) ↦ ρ^r_0 a + ρ^r_1 b`
     is invertible on F128².
   - The fold becomes `y_1'^r = y_1^r + W^r R^r`, uniform on F256^L.
   - P sends `T'^r = ⟨b(r̄^r, ·), y_1'^r⟩` and `ybar^r = Σ_l eq(r̄^r, l) pad_l + W^r pad_{R^r}` (`t_pad` F256 values,
     `zk::ybar`).
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

**Assumptions.** The argument above is exact for uniform prover randomness. In the implementation:
- **ChaCha20 is a PRG.** Every pad, mask, padding coefficient, extra lane and leaf salt is ChaCha20 output under the
  session's 32-byte seed from the OS (`ProverRng`, `LeafSalts`), so the simulation is computationally indistinguishable,
  with ChaCha20's distinguishing advantage as the gap. The seed's quality is trusted.
- **hm96-sha512 leaves hide.** Caps, paths and unopened leaves are hm96-sha512 outputs over 192-byte salts: hiding in
  the random-oracle model (soundness ASSUMPTIONS.md §6, §7), binding under SHA-512 collision resistance.
- So the notion is computational special honest-verifier zero knowledge for one session.
- **Refused coins.** The prover's refusals are exactly the ones the argument uses: `β = 0` (the inner proof),
  `ρ^r_0, ρ^r_1` dependent over F128 (`W^r` invertible), and mask-slot weights of rank `< 128·(claim points)` (§6).
  The mask slot must have `blocks · 2^(mask_log−7) ≥ 128 · 4` words for the rank to be reachable; RoPE has 1024.

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

**The audit** (`zkaudit --zk`). Every byte the verifier keeps, and the proofs' opened data (with `final_c'`, the opening's
batching nonces and the commitment's parameters), is attributed to a class:
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

## 9. What M2 still needs

- **The coin commitment** (DESIGN.md §11.1). One up-front commitment per session: a SHA-512 Merkle root over
  Halevi–Micali commitments to every coin, one per (table, repetition, round), each stream a subtree. Every prover
  machine checks each coin's opening before using it. This replaces M0's `coin_seed` step 0 (a seed commitment with
  `derive()` streams, which needs a PRF assumption). The prover's refusals (`β = 0`, mask rank, `ρ_0, ρ_1` dependent)
  stay functions of committed coins.
- **The single-rewind simulator** (Goldreich–Kahan). The simulator runs the honest prover on a dummy witness until the
  final openings reveal the committed coins. The dummy witness must carry the public region words (`b ‖ c` and the
  declared outputs), every private word zero and the mask slot uniform: its region claims' `s_hat_v` are sent in the clear
  before any coin opens, and a verifier compares them with the public data. It then rewinds once to just after the coin commitment and runs §8's
  per-table simulator on those coins, for every table, with independent masks. Every message before the last coin is
  masked, as §7 now makes Ligerito's too. So the dummy run is indistinguishable from a real one until the openings, as
  §11.1 requires.
- **C1's level-0 root check**, kept under M1. M0's commit hook compares rep 1's level-0 root with rep 0's and stops
  before rep 1 opens anything, and the device refuses salt stream 0 for any tree whose leaf count is not level 0's
  (commit `e2190ca3`). M1 keeps it:
  - both reps prove one witness, uniform mask slot included;
  - §7's padding and extra lanes are drawn once per session and belong to level 0's one tree;
  - the pads trees take `next_id() ≥ 1`.
- **GPU.** `--zk` is CPU-only. The masking challenger and the inner proof are host-side and small. The mask slot fill,
  §7's padding and extra lanes, and `commit_zk` belong on the device, expanded from ChaCha20 like the hm96 salts. The
  device prover also needs §7's fold and messages.
- **Review.** A red team on this document, then the Lean reference prover and its differential test, before any
  `COMPLETE_*` class.
