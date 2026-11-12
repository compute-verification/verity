# verity/flock-circuit: Flock ZK, its masking and its coin tree

What `flock-circuit --zk` sends, where every mask comes from, and what the verifier checks. This is the layout the Lean
reference prover must reproduce byte for byte (soundness DESIGN.md §6: a deterministic function of statement, witness,
prover seed and verifier coins), and the statement its zero-knowledge theorem is about.

Code:
- `src/zk_veil.rs`: masking, replay, constraints, the inner proof, the simulator's pieces.
- `src/bin/flock-circuit.rs`: `prove_cpu_zk`, `zk_constraints`, `verify_zk_with`, `simulate_full`, `zkstat`, `zkaudit`; the coin tree's
  `VStar`, `gk_simulate`, `rewind`, `zkrewind`.
- `src/coin_tree.rs` (the coin tree): the verifier's coin commitment and the prover's check of every opening.
- `../flock-zk-b684b12.patch`: Ligerito level 0's hiding in upstream Flock (`pcs::ligerito::zk`, `commit::commit_zk`, the F256
  split prover and verifier), and the CPU prover's 64-bit nonzero offsets (`lincheck::CscCircuit`, `r1cs::flatten_csr`), so
  one matrix may hold more than 2^32 nonzeros. The GPU prover's are 32-bit: `--gpu` refuses a circuit past them
  (`circuit::GPU_NNZ_MAX`).

**Claim.** Every message a session sends is masked, committed or public, and the audit (`zkaudit`) attributes every
byte. The target notion is VEIL's special honest-verifier zero knowledge for the whole session (both repetitions),
meaning one simulator for every fixed coin sequence (DESIGN.md §11), statistical in the common-reference-string model
under the assumptions of §7. Malicious-verifier zero knowledge is the coin tree's, from its coin commitment and single rewind (§9).
The prover refuses the degenerate coins (`β = 0`, a mask slot of deficient rank, dependent `ρ`, a zerocheck outer coin
`r_j = 1`) as they open, before it sends anything that depends on them. Each rep has its own pair of level 0's extra
lanes (§7). The proof class stays `NON_ZK_PROOF` until the Lean proof covers it.

Conventions. The field is GF(2^128) (`F128`, characteristic 2, so `+` is `−`). F256 is `F128 ⊕ F128·U`, written by
coordinates `(c0, c1)`. An F128's bytes are `lo` then `hi`, each little-endian. `v'` is a sent (masked) value, `v` the
value upstream's prover computed, `h` a pad, and `v' = v + h`. `m` is the dense variable count and `k_log` the block size
(RoPE: `m = 25..27`, `k_log = 22`). Ligerito at `m = 25`: `initial_k = 6` (64 lanes), `L = 2^12` rows per lane, rates
`1/2, 1/4, 1/8, 1/16`, queries `277, 106, 71, 53` (level 0's count is §7's, above Fast100's 218), OOD samples `1, 2, 2, 2`.

## 1. Order of one repetition

Everything below is on the rep's live stream (`circuit/rep{r}`), in this order. Items marked ZK are Flock ZK's; the rest is
Flock's circuit prover and upstream Flock.

~~~text
bind_statement      label flock-r1cs-v0 · bytes statement digest · bytes witness cap (level-0 tree, 68 lanes)
zerocheck           label flock-zerocheck-v0
  ZK                label verity-flock-zk-pads-v0 · bytes pads root (64)
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
  ZK                label verity-flock-zk-l0-open-v0                      (level 0's masking begins)
                    per OOD sample: squeeze z (log_n) · scalar y' · bytes nonce · squeeze β
                    msg_0' (u_0 slice 2, u_2 slice 2)
                    per j < initial_k: squeeze r_j (F256) · msg_{j+1}' for j+1 < initial_k
  ZK                slice e_0' (2) · slice e_1' (2) (rep r's pair) · squeeze ρ_0, ρ_1 (F256)
  ZK                label verity-flock-zk-l0-close-v0                     (level 0's masking ends)
  ZK                slice T' (2) · slice ybar (2 t_pad)
                    code-switch message · cap_1 · ... levels 1+, unchanged ...
ZK inner proof      label verity-flock-zk-inner-v0 · scalar final_c' · squeeze ε · slice [ρ, σ] · squeeze γ (10)
  ZK                bytes τ's commitment (64) · squeeze β · slice Y (s+192) · squeeze queries (168)
proof bytes         bincode (Commitment, R1csProofLigerito with the masked fields, InnerProof)
~~~

Upstream never absorbs `final_c'`; Flock ZK absorbs it first in the inner proof (§5). The claim order in the opening is ab, then
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

**Randomness.** Everything comes from `ProverRng`, and in a proving build every sample is the OS's own bytes (`getrandom`),
read where it is sampled. `f128s(stream, index, n)` reads `16n` fresh bytes, whatever its
address, so each value below is sampled once and kept where it is used again: level 0's hiding for the table's session (the
device's commit and the ZK server's recheck take the same), each rep's pads in the rep's state, and each block's mask words
with the table's witness. Under the outer ZK server (§10.4) Lean reads all of them from the OS: each rep's frame in the
Lean session loop samples the rep's `h`, `mu`, `pad_h`, `pad_mu`, `τ`'s salt and the pads tree's salts
(`Flock.Firewall.Frame.Draws`, in the layout below with the same two products) and keeps them, the prover getting only the
pads tree's root; the loop samples each table's mask-slot words, which it writes into the witness it commits, and the
table's level-0 hiding and salts, which the prover takes in place of its own (§10.2); the Lean finish samples each rep's
salts for levels ≥ 1; and `flock-firewall tap` serves every other read (every other tree's salts:
`zk_hooks::set_os_source`). The coin tree's nonce is the loop's, sampled with the `Hello` it sends (§10.4).
The addresses matter only under a test harness's injected seed (feature `seed-injection`), where
`ProverRng` is ChaCha20 keyed by that seed and `f128s` reads `16n` bytes from byte offset `32·index` of stream `stream`, so
the harness's proofs repeat. With `base = rep << 32`:

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

Streams 1–3 belong to Flock's circuit prover: mask, padding, and the injected-seed salt tape. None of this is derived from the transcript.

**Several tables** (`--session-tables J`, the CPU prover). Every table's samples are OS reads of its own
(`ProverRng::for_table`; under an injected seed, table `j` reads every stream above at ChaCha20 stream `purpose | j << 32`).
Its salt trees are ids `j << 32` (its level-0 tree) and up, below the next table's (`LeafSalts::table`, every tree's salts
read from the OS). So no two tables share a pad, a mask word, a padding coefficient or a salt. Table 0's addresses are a
one-table session's.

## 3. The pads commitment

The code is Reed–Solomon over GF(2^128). The message is `(h ‖ pad_h)` of length `L_h = s + 192`, read as coefficients
of `Σ m_i X^i`. The code is evaluated at the `N = 8 · next_pow2(L_h)` points `x_j = F128(lo = j + 1, hi = 0)`, which
gives RoPE `N = 4096`. There are two rows, `C(h ‖ pad_h)` and `C(mu ‖ pad_mu)`. Leaf `j` is the `hm96-sha512/v1` leaf of
`x = SHA-512(bytes(C_h[j]) ‖ bytes(C_mu[j]))`. Its salt is leaf `j` of a fresh tree from the proof's `LeafSalts`: 192
bytes read from the OS (`getrandom`), as every leaf salt is. The tree's id comes from `next_id()`, so it is never 0 and never
reused. Nodes are `SHA-512(left ‖ right)`. The 64-byte root is the item after `verity-flock-zk-pads-v0`. Under the ZK server
the rep's Lean frame samples the leaf salts and computes the root itself (`Flock.Firewall.Frame.padsRoot`), and the prover's
tree over the frame's samples must have that root.

## 4. The constraints

The verifier re-runs upstream's zerocheck and lincheck verifiers on the masked proof (`zk_veil::replay`), sampling the
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
   `hm96::leaf(SHA-512("verity/flock-zk/tau/v1\0" ‖ τ), salt)` with a fresh 192-byte salt (`tau_commitment`, §9).
3. `β` ← V. P refuses `β = 0`. P sends `Y = (h + β mu ‖ pad_h + β pad_mu)`.
4. 168 query words ← V, reduced to `j = lo mod N`. P opens `(C_h[j], C_mu[j])` with salts and paths. The proof carries
   `τ` and its salt.
5. V checks that `τ` opens its commitment, `⟨c, Y[..s]⟩ = t + β τ`, `C(Y)_j = C_h[j] + β C_mu[j]` at every query, and each
   path to the pads root.

`τ` in the clear before `β` (masking's first version) would let a verifier test `⟨c, Y⟩ = t + βτ` as soon as `Y` arrives,
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
opening's label, before any `s_hat_v` leaves (`mask_rank_ok`, under 2 ms: an echelon basis that reads mask words only
until its rank is full), and stops otherwise. RoPE: rank 512 needed, from
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
   - V samples `ρ^r_0, ρ^r_1 ∈ F256`. P refuses them unless they are independent over F128, so `W^r : (a, b) ↦ ρ^r_0 a + ρ^r_1 b`
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
   identity holds at every position, as an identity of codewords. `Enc(c·ybar^r ‖ ybar^r)` is evaluated at the queries alone
   (`zk::pad_codeword_at`): its message is nonzero on `2 t_pad` entries, so a query's value needs only the transform's last
   `⌈log₂ t_pad⌉` layers within its block, never the whole length-`2L` codeword (`zk::pad_codeword`, the test's reference).
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

   At `m = 25..27` the counts are computed with 64 more padding symbols; elsewhere they are the least count. `--zk` refuses `m ≤ 24`, where no count reaches 100 bits at rate 1/2, and any
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
spent twice. A pair shared by the two reps would be: `(y_1'^0, ybar^0)` and `(y_1'^1, ybar^1)` would both sample on one
`(R, pad_R)`, and `N·(·)^0 + (·)^1` with `N = W^1 (W^0)^-1` eliminates it: a witness
functional at every level-0 position either rep opens and at the level-1 positions both open. With `k` extra lanes
shared by both reps, joint uniformity needs the `4 × k` matrix `[W^0; W^1]` to have rank 4, so a pair per rep (4 lanes)
is the minimum.

**Checked:**
- `zk_veil` test `exact::a_pair_per_rep_hides_the_witness_and_a_shared_pair_does_not`: at toy size (4 witness lanes of 16
  words, 3 queries per rep, 5 distinct positions), with the protocol's own `lane_codeword`, `pad_codeword_at`, `ybar` and
  `enforced_extra`. For fixed coins the view is F128-linear in the witness and the randomness, and it hides the witness
  exactly when the witness's images lie in the span of the randomness's. Even with all of both `y_1'^r` revealed, the
  per-rep pairs hide it; a shared pair does not, nor does `t_pad` below the distinct opened positions. The verifier's row
  check holds on the model.
- selftest `zk_session_relations_vary_with_the_seed` (RoPE `m = 25`, one coin seed): the finding's two functionals
  (`session_relations`: 521 F256 values at level 0, 42 at level 1) differ in every word between two prover seeds. A
  prover that folds rep 0's pair in both reps (rejected by the verifier) reproduces the leak: every word is equal. The
  level-1 functional exists only at the positions both reps open at level 1, which the coins select for each rep
  independently. On some statements the reps share none: RMSNormTriton_v5 at N 2048 shares none under any of 16 tapes,
  and then there is no level-1 relation to leak. So the case checks level 1 at every shared position, and it requires:
  - that each rep opened level-1 positions, so an empty level-1 functional is a disjoint sample, not a lost record;
  - that the level-1 functional is non-empty whenever a position is shared;
  - that both sessions of each kind sampled the same positions;
  - that level 0 is non-empty.
- The statistics (§8) test both functionals as classes of one value per session.

**Assumptions.** The table's argument is exact (perfect hiding) for uniform prover randomness, given the commitments hide.
- **hm96-sha512 hiding is statistical, in the standard model** (`verity/commitments/hm96/PROTOCOL.md` §3: a
  leftover-hash-lemma argument over the Hankel family, which uses no property of SHA-512). Caps, paths and roots are
  functions of the leaves, so they inherit the bound. It needs:
  - uniform, independent salts, never reused: every leaf's 192 bytes are read from the OS (`getrandom`), each tree's once
    (`LeafSalts`: level 0's are kept for the session, and a tree whose salts the ZK server's Lean finish sampled is kept until the device's tree takes them), and
    `next_id()` is never 0 and never repeats (level 0 is the one tree both reps share, each leaf salted once); the pads
    trees and τ's commitment have their own salts;
  - the pinned key outside the `2^-64` fraction of bad keys: the common-reference-string step under `hash-derived-key`.
    The distance is then about `N·2^-192`, with `N` every leaf ever committed under the key (an `m = 25` session commits
    about 2^14.4), so a lifetime of `2^64` leaves stays within `2^-128`.
- **So masking is statistical special honest-verifier zero knowledge for one session**, in the common-reference-string model,
  with distance about `N·2^-192`. The claim ids are `statistical-hiding`, `common-reference-string` and `hash-derived-key`.
  No random oracle is involved.
- **No premise on a stream generator.** Every pad, mask, padding coefficient and extra lane (`ProverRng`) and every leaf
  salt (`LeafSalts`) is the OS's own bytes, so the claim is statistical with the OS
  generator's output as the uniform prover randomness it assumes, and nothing rests on ChaCha20 as a PRG. (A test
  harness's injected seed expands all of them with ChaCha20, so its proofs repeat; a proving build has no injected seed.)
- **Collision resistance** (`cr/sha-512`) enters only soundness (binding of the prover's trees) and the coin tree's binding of the
  verifier's coin commitment (§9), not masking's hiding.
- **Refused coins.** The prover's refusals are exactly the ones the argument uses: `β = 0` (the inner proof),
  `ρ^r_0, ρ^r_1` dependent over F128 (`W^r` invertible), mask-slot weights of rank `< 128·(claim points)` (§6), and a
  zerocheck outer coin `r_j = 1` (the replay's round `j` divides by zero there, and the honest pads miss the inner proof's
  constraint by a witness-dependent amount). Each stops the prover before it sends anything that depends on the coins it
  reads, and reads only opened coins; `r_j = 1` is checked as the coin opens, in `zk_veil::Masking`'s sample. Its
  completeness loss: under uniform coins the honest prover refuses with probability at most `(#outer coins)·2^-128` per
  rep, `m − 13` outer coins. §5's self-check refuses no coin: an honest witness passes it for every `γ`.
  The mask slot must have `blocks · 2^(mask_log−7) ≥ 128 · 4` words for the rank to be reachable. Both sides refuse a
  statement, or a table of a split session, with fewer than `128 · 4 + 64` words when it is built, before `Hello`
  (`zk_shape_ok`). RoPE and GEMM have 1024 at every split tried: m ≥ 25 at their `k_log` means at least 8 blocks.
  Both sides also refuse a mask slot below `2^13` bits (`slot_log < 13`, `SLOT_LOG_MIN`): the bound on this check
  refusing an honest prover's coin covers no smaller slot. Every deployed circuit has `mask_log = 14`.

Negatives, each rejected on both reps (RoPE CPU selftest):
- another extra lane folded than the committed one;
- `T'` forged;
- `ybar` forged;
- `e` forged with an honest `T'`. Ligerito accepts this one, and only the inner proof's lane-phase rows reject it.

## 8. The simulator, the statistics and the audit

**The simulator** (`simulate_full`, per rep, fixed coins; both reps from one dummy session, so the session's structure is
the dummy's with the dummy's randomness). Its input is the public statement, the verifier's coins for the rep's stream
(every squeeze, in order) and the two link points. It builds:

1. uniform exposed values and uniform masked-claim `s_hat_v`;
2. the region claims' `s_hat_v` from public data;
3. the opening as the honest prover makes it on a dummy (all-zero) witness with fresh masks, against the same coins, with
   level 0's exposed values sampled uniformly. This is Goldreich–Kahan's dummy-witness run, restricted to the part whose
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
prover randomness zero) must be distinguished, and every simulated proof accepted.

**The audit** (`zkaudit --zk`). Every byte the verifier keeps, and the proofs' opened data (with the opening's batching
nonces and the commitment's parameters), is attributed to a class. The inner proof's round values are `final_c'`, `[ρ, σ]`
and `Y`, in that order (selftest `zk_audit_attributes_the_inner_proof`). The audit checks that:
- masked values differ from the control's;
- region `s_hat_v` equal their public recomputation;
- every Ligerito value differs from the control's;
- nothing is unattributed or in the clear (`attributed: true`).

This is attribution, not hiding: that a mask is applied, not that it hides. Hiding is §7's argument, its exact check and
the statistics' relation classes.

The prover's own replay takes its lincheck's folded `comb` as an argument (`LincheckProof::comb_partial`, never
serialized), never from a proof.

## 9. The coin tree: the verifier's coin commitment and the single-rewind simulator

The design for malicious-verifier zero knowledge for one batched session, in the model of ASSUMPTIONS.md §2 and §6 and
DESIGN.md §11.1: the verifier commits to every coin before the prover's first message, then opens them round
by round, and a Goldreich–Kahan simulator rewinds once, to just after that commitment. The prover refuses the
degenerate coins as they open (the Claim above).

**The coin commitment** (`src/coin_tree.rs`, the keyed coin tree `coin-tree/hm96-sha512/v2`,
`SessionConfig::coin_tree`; the verifier's side is `Flock.CoinTree`).
- The schedule is pinned in `Hello` (`coins: {block, nonce, rounds_log, scheme, streams}`): streams `session`,
  `circuit/rep0`, `circuit/rep1`; `2^9` rounds per stream; blocks of 512 coins. A RoPE rep uses 100 rounds and at most
  277 coins in one (level 0's queries); the session stream's round 0 is the two link points.
- The tree is keyed to the session: the prover's `Hello` carries a fresh 32-byte nonce `ν`, and every SHA-512 input of
  the tree is prefixed by it, `H_ν(m) = SHA-512(ν ‖ m)`.
- At `Hello` the verifier samples a fresh 256-byte hm96 key `K` for this session's tree (never reused, never revealed
  before its answer), then every block and a 192-byte salt `y` per (stream, round), and commits: per block the chain
  `T_512 = H_ν(TAIL)`, `T_i = H_ν(CHAIN ‖ c_i ‖ T_{i+1})`; the leaf's digest
  `x = H_ν(ROUND ‖ u32be(stream) ‖ u32be(round) ‖ T_0)` and the keyed hm96 leaf `H_ν(LEAF_PREFIX(K) ‖ x ⊕ M(K)·y ‖ c)`
  with `c = H_ν(SALT_PREFIX ‖ y)`; each stream's leaves under a SHA-512 tree (a subtree), and the streams' roots under
  one more, padded with `H_ν(PAD)`. The 64-byte root and the key are `Hello`'s answer, before the prover's first message
  (`Commit`).
- Under `serve --tree-from HOST:PORT` (`src/tree_from.rs`) the challenger, on a host the developer doesn't control, samples
  the tree instead, under the prover's nonce, and opens each round itself after what it follows: the link points after
  `SHA-512(root_F ‖ each root_B in table-name order)`, every other round after its message's SHA-512 (the record's
  `msg_sha512`). The challenger checks every opening against the tree before it answers, so the record is the one a tree
  sampled here would give; a refused, broken or bad opening ends the challenger's session, and no coin is flipped again.
- Every answer is the round's `n` coins, `T_n`, the salt and the path (`Spec::opening_words(n)`), so an opening reveals only
  the coins the round uses. The prover checks every opening against the root before it uses a coin (`coin_tree::Checked`,
  or under the ZK server the Lean session loop, §10.4) and stops at one that does not open, before it answers anything that
  depends on it.
- This replaces Flock's circuit prover's step 0 (a seed commitment with derived streams): no PRF. The prover's refusals (`β = 0`, the mask
  rank, dependent `ρ`, an outer coin `r_j = 1`) stay functions of committed coins.
- **Two properties, two roles.** *Hiding* of the unopened coins is what soundness uses: statistical in the standard model
  (hm96 `PROTOCOL.md` §3), since every block and salt comes from `/dev/urandom`; no random oracle. *Binding* is the one
  assumption the coin tree's zero knowledge adds: collision resistance of `x ↦ SHA-512(ν ‖ x)` under the session's nonce
  (`cr/sha-512`), so a collision in V*'s auxiliary input, made before `ν` existed, opens nothing. It holds whatever
  salts V* chooses (hm96 §4 holds for every key), so a rewound V* cannot open another coin. Against quantum verifiers
  the commitment must be collapse-binding (ASSUMPTIONS.md §6).

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
   lanes and salts, against a local challenger that commits to those coins; no interaction with V*). So the rewinds are
   independent trials, each succeeding with V*'s completion probability `p`. Each rewind is V* from the start of the
   session, answered by a scripted prover whose messages are the per-rep SHVZK simulations (`simulate_full` on that fresh
   opening, §8, each rep with its own pair). A `RecordingChallenger` recovers each round's message by replaying the
   verifier on the simulated proof. Every coin V* answers is checked against its commitment. The first rewind that
   reaches the proofs is the output. A rewound V* that opens another coin than it did in step 2 is a binding failure, and
   the simulator fails; so does exhausting the bound.

**The parameter `t` grows with the security parameter:** `t = Θ(λ)`, so that `p̃ ∈ [p/2, 2p]` except with probability
`2^-Ω(λ)`, and then all `4t/p̃ ≥ 2t/p` rewinds fail with probability at most `e^-2t`. A fixed `t` would leave a failure
probability that is noticeable against some V*. The tests run at `t = 4` (`zkrewind --gk-t`); that
is a test's λ, not a bound for any served claim.

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
one message hashes "good" (half the time) and otherwise with probability `ε = 1/16`: `cap-gated` on the witness cap,
`late-gated` on rep 0's round with `T'` and `ybar`. A simulator that reused messages across rewinds
fails against the gated ones. `zkrewind --zk --n N` prints N real and N simulated views per strategy, and
`tests/zk_rewind_stats.py` compares how they end (a permutation test), their verdicts, rep 0's first 256 masked values,
every received proof's checked fields and the session relations of §7.

**The rewinds are independent trials** (`zkgk --zk --attempts A --reuse-dummy`): for each gated V*, conditioned on an
extraction run whose gate was bad, A rewinds with fresh dummy openings complete at the rate of A real sessions (a Fisher
test in `zk_rewind_stats.py`), while A rewinds on the extraction run's own dummy opening (the defect a reusing simulator has)
complete at about ε, which shows the test's power.

Selftest negatives: `zk_verifier_coin_off_its_commitment` (the prover stops at rep 0's round 3),
`zk_verifier_commits_beta_zero` (it stops at the β round, 98), `zk_verifier_commits_outer_coin_one` (it stops at the
round that opens the outer coins), `zk_dummy_witness_passes_the_region_claims` (the dummy is
rejected by the inner proof alone), `zk_failing_witness_stops_before_tau` (an honest plan's prover on a witness with a
flipped unit bit stops after the batching coins: rep 0's stream ends with `[ρ, σ]`, and no `τ` commitment, `β` or proof
goes out).

**Level 0's root check (C1).** Both reps prove one witness, level 0's randomness is sampled once per session and belongs
to its one tree, and the pads trees take `next_id() ≥ 1`.

**Several tables per session** (the public-circuit ZK proof's §2.3) are on the CPU prover, `--session-tables J`; the
device prover proves one table per session and refuses J above 1:
- the sample's units split into J equal tables `circuit-00`, `circuit-01`, ... (`session_tables`), under Σ over every
  table's Σ. The server refuses at configuration a Bernoulli sampling design (its count is random), a count J does not
  divide, and tables `--zk` cannot prove. The session's sampling design is the record's `unit_draw`: each table's
  header carries only its part as a subset sample;
- randomness per table (§2) and a rank check per table (§6). Per table is the joint check here, because no claim or
  message reads two tables' witnesses or masks. The tables' witnesses may be correlated: two sampled proof blocks in different
  tables can read the same row;
- one `Commit` carrying every table's level-0 root: a later table commits its root before any stream opens, and its reps
  stop unless they bind it;
- a coin subtree per table and rep;
- every proof after the session's last coin (the barrier).

The selftests are `tables_two_in_one_session`, `tables_proofs_before_the_last_coin_detected` (a prover without the
barrier is accepted by the verifier and caught by the order check), `zk_tables_share_no_randomness` and
`tables_unsplittable_draw_refused`.

**The simulator over several tables.** `zkrewind` and `zkgk` take `--session-tables J`: `gk_simulate` extracts,
simulates and replays every table's streams in the prover's order. `zk_rewind_stats.py` tests each table's proofs and
relations as their own classes, and fails any value that table j shares with table 0 in every view (`across_tables`).

## 10. The ZK server's send check: nothing leaves the GPU that the ZK server did not fix

In a recursive session the developer runs a **ZK server** it trusts between its workers, which nobody trusts, and the
auditor (the challenger and the verifier). The workers may run any kernel; the question is what they can make leave. A
**free choice** is any bit a worker can vary that reaches the auditor, or the proof bytes, without the ZK server having either
computed it itself or held it to a relation that leaves the worker nothing to choose. The send check is the rule, per
message kind, by which the ZK server removes every free choice.

Where the boundary runs:
- **Inner phase** (`rec/inner-transcript/v1`, ZK off): the whole circuit prover is the worker, and `verity_flock.rec_live`'s
  proxy is the ZK server, its send check Lean's `flock-firewall live` (`rec_live --lean`). The challenger receives only the
  public schedule and salted hashes (§10.3).
- **Outer phase** (`flock-circuit prove --zk --gpu`): the device (`flock_cuda_prove_circuit`) is the worker; the host side
  of `flock-circuit` (the masking challenger, the live stream, the send check, and the session's Lean finish it relays) is the ZK server. The
  device reaches the ZK server through three channels: each round's frames (`rounds(content, n)`, at its squeeze), the
  level-0 root (`hook(root)`, at the session's `Commit`), and the returned `(R1csProofLigerito, Commitment)`. The ZK server
  checks all three. The device holds the witness, the level-0 hiding (`Level0Zk`: padding and extra lanes) and the salts of
  the trees it builds (level 0's and levels 1+'s, which the host hands it tree by tree), but never the pads `h`, `mu`, their
  padding, the pads tree's salts or the inner proof's randomness, which the rep's Lean frame samples and keeps (§10.4:
  neither the device nor `flock-circuit`'s host side receives them). Holding the salts it does fixes nothing it sends:
  every hiding leaf and path that leaves is the recheck's or a recomputed tree's (§10.1), and the pads tree is the ZK
  server's own.

### 10.1 The rule, outer phase (one rep, in §1's order)

Five kinds of answer: **constant** (the ZK server holds the only allowed value), **recomputed** (the ZK server computes the value
from the host witness, its own randomness and the coins, and refuses any other), **masked** (the ZK server adds a pad the device
never sees), **the ZK server's own** (the device never touches it), and **relation** (a check that fixes nothing by itself,
kept as an early tripwire).

| item (§1) | rule | why the device has no free choice |
|---|---|---|
| the frame skeleton: domain, every label, each item's op, kind and length, every round's coin count | constant | a function of the statement's shape; the ZK server refuses any other label or item |
| statement digest (`bind_statement`) | constant | the digest of the statement the ZK server loaded |
| witness cap: `Commit`'s root, `bind_statement`'s cap, Ligerito level 0's `bytes cap` | recomputed | the level-0 recheck (§10.2): the Lean verifier's commitment of the host witness under the session's level-0 salts and hiding |
| pads label and root | the ZK server's own | the Lean frame samples the pads and the pads tree's salts, computes the root, and inserts both after the zerocheck's label; only the root reaches the prover |
| masked algebraic values: `round1_ab'`, `round1_c'`, `G_j`, `final_a'`, `final_b'`, `q_j`, `z_partial'` | masked | the wire carries `v + h` with `h` uniform and the Lean frame's alone, so it is uniform whatever `v` is, and a worker, which never holds `h`, cannot pick what leaves. `v` enters the view only through §4's rows, which the Lean finish evaluates on the pads at the rep's end; a `v` that breaks a row stops the session at its `Finish` |
| `final_c'` | the ZK server's own | the inner proof sends it, from the host |
| `s_hat_v` of the masked claims (ab, c) | recomputed | `Σ_x eq(p_w, x) z[128x + v]` over the host witness (§6), whose mask slot holds the words the Lean session loop sampled and wrote into it (§10.2), so no worker picks the words that hide it; the claim points are coins |
| `s_hat_v` of the region claims | recomputed | `Stmt::region_s_hat_v`, from public data |
| Ligerito's `target` | recomputed | a function of the `s_hat_v` and the opening's coins (`r''`, `γ`) |
| level 0's OOD value `y'`, lane messages `msg_j'`, `e^r'` | masked | as the algebraic values (`lg_ood`, `lg_msg`, `lg_e`) |
| every grinding nonce | constant `0` | live coins need no grinding (the device's live challenger absorbs nonce 0) |
| `T'^r` | recomputed | `T'^r = ⟨w^r, y_1'^r⟩ = ⟨w^r, y_1^r⟩ + ρ^r_0 e^r_0 + ρ^r_1 e^r_1` with the honest `e^r = ⟨w^r, R^r⟩`: `w^r` from the coins and claims, `y_1^r` the fold of the host witness's lanes, `R^r` the host's extra lanes. The lane-phase relation alone (`claim_k + ρe + T'`, on the device's unmasked values) does not fix it: a device that shifts `e` by `δ` and `T'` by `ρδ` keeps the relation and sends 256 bits of its choice |
| `ybar^r` | recomputed | `zk::ybar` from the session's level-0 padding (host), `w^r` and `ρ^r` |
| levels 1+: the code-switch message, every cap, OOD value, sumcheck message and nonce | recomputed | functions of `y_1'^r`, the coins and the salts the host expands for trees `1..R` |
| the inner proof: `[ρ, σ]`, `τ`'s commitment, `Y`, its openings | the ZK server's own | `[ρ, σ]` and `Y`: the rep's Lean frame computes them from its samples and the coins and refuses an inner round that is not them (`innerOwn`); the rest the rep's proof file, which the Lean finish computes (§10.4), from the pads on the host, over §4's rows replayed from the passed values with the lincheck's comb the ZK server folds itself: a device's comb would choose the rows the inner proof proves, so the ZK server drops it (the outer proof never carries it; the verifier folds its own) |
| the proof bytes | recomputed | every transcript field equals what the ZK server passed; level 0's rows and paths are the recheck tree's at the queries the coins fix; levels 1+'s are the recomputed trees' |

So the rule is: **the ZK server masks the algebraic phase and level 0's lane rounds, and recomputes the polynomial
commitment's outputs** (the level-0 commitment, the opening's `s_hat_v` and `target`, and everything from `T'` on), and
folds the lincheck's comb for its own relation check, as the verifier does. It does not recompute the zerocheck or the
lincheck, the bulk of the device's arithmetic. What it recomputes is a CPU Ligerito: the commitment over the whole
witness, and the opening's basis fold `w^r`, which `T'`, `ybar` and levels 1+ all read.

**The pass decision is the Lean frame's.** Each device round goes, unmasked, to the rep's frame in the Lean session loop
(§10.4), which holds it to the statement's round in its place: in the algebraic phase the statement's scheduled round with
every value `0` (`Skeleton.rounds`, on the table's level 0), in the opening the loop's own opening round (`Drive.openStep`).
The device round passes when it is that round byte for byte but for the values the algebraic phase masks
(`Flock.Firewall.Frame.check`): the same frames, labels, bytes, headers, padding, kinds, lengths and coin count in the
algebraic phase, and every byte in the opening, level 0's values included (`live/src/pass_vectors.json`,
`tests/test_firewall_frame.py`). Rust keeps no schedule or round check of its own: it asks the loop each round's place and
sends the device's round there.

**A refusal stops the session at its `Finish`, never before.** When a check fails mid-session (a round off the statement's
in its place, the device's level-0 root, the end check on the returned proof, a row that fails on the pads), the
ZK server keeps the reason and keeps releasing the statement's schedule in the device's place, and nothing of the refused
item leaves:
- in the algebraic phase, the statement's next round with every value `0`, which leaves masked as its pads alone
  (`Flock.Firewall.Frame` zeroes every algebraic value of a stopped rep, whatever round it is given);
- in the opening, the loop's own opening round, the ZK server's computation (an honest device's round);
- at `Commit`, the recheck's root;
- for the returned proof, the ZK server's own (the passed values and the Lean finish's opening, as on any rep), and the inner
  proof speaks on schedule without its self-check (§5), as masking's simulator's does.

A device that sends more rounds than the statement has gets coins back and nothing leaves; one that returns early has the
rest of the schedule released by the ZK server itself. A failed rep's rounds from its failure on reach the loop as no
device round (the place's squeeze alone), and every rep of the session stops at its first failure: from then on the
loop hands no rep's frame a device round, whatever the device sent (`Flock.Firewall.Frames.Sessions.stopped`), so each
frame fails each of its rounds, releasing the statement's round in its place. Then the session's `Finish` goes
without its proofs, and the auditor
sees the stop there. What leaves before `Finish` has the form and the distribution of an honest session's, wherever the
check failed: under pads `h + v` a stopped rep releases byte for byte what the honest rep with values `v` releases under `h`
(`tests/test_firewall_frame.py`). So a stop the ZK server makes reveals one bit a session (stop at `Finish` or not), and
`log₂ Σ_{i≤b} C(m, i)` bits over `m` sessions with at most `b` stopped (the claims' `ZkServer.zk_deferred` and
`stream_leak_deferred`). Where the release ends is the prover's choice, though: the Lean loop
(`Flock.Firewall.Session.relay`) ends it where the prover closes its connection or sends a frame the loop can't read, and
closes the auditor's connection there, so the auditor sees how many of the session's frames reached it. A release therefore
also leaks its end position: a session of `M` messages has up to `M + 1` stops beside its finish, `log₂(M + 2)` bits rather
than one. It is counted by position, as `Flock.Guarantees.RecursiveZKLoopAnySum` counts a recursive session's stops
(`1 + Σ_p (M_p + 1)`, `LoopAny.card_loopStops`), and the bound that holds at a prover that cuts is §10.5's (its table's first
row a recursive session, and its bound over a stream). A ZK server that refuses is evidence against its rack. It
never stops serving for good after failed sessions: a worker chooses its sessions' verdicts, so the leak is bounded per
outcome of each session, and is not capped over a monitoring period (`Flock.Guarantees.SessionLoopZKAnyDeployed`).

What the deferral costs: a bad worker is caught only at its session's `Finish`, one session's latency later than at its
first bad message, and the rest of a doomed session's device and ZK server work is spent for nothing (the ZK server's share
is the Lean finish's opening and inner proof, which it runs on every session anyway). Timing (when a message leaves) is not in
this rule; the network certifier covers it.

**A worker that fails is a failed check like any other.** The device runs on a thread of its own (`firewall::run_worker`),
and the ZK server serves its rounds and its `Commit` on the session's thread, so where a rep ends never depends on its worker.
Each way the worker can fail fails the rep (`FIREWALL-REFUSED: the device failed: …`):
- an error it returns (a nonzero return code, its return code 240's claim of degenerate coins among them);
- output that does not decode;
- a panic;
- silence past the ZK server's deadline (`FC_FIREWALL_DEADLINE_S` seconds between two of its asks, an hour by default).

The ZK server then releases the rest of the schedule itself, takes the `Commit` with the recheck's root if the worker never
reached it, and stops at `Finish`. A worker the ZK server gave up may still run, but nothing it asks later is answered. A tree
the device samples at another size than the Lean finish's gets salts of its own (`LeafSalts` keeps the finish's salts by tree
id and size), so its opening fails the check, and nothing panics.

The stops the coins decide are the ZK server's own, taken from the public coins and never from the worker:
- a zerocheck outer coin 1: the rep's Lean frame, from the coins the stream answered (`Flock.Firewall.Frame.Masker.answer`);
- the mask slot's rank at the opening: the Lean frame, over the claim points it computes from the table's coins
  (`claimPoints`, `maskRank`);
- level 0's `ρ_0, ρ_1` dependent over F128: the Lean frame, from the four words the stream answered after level 0's last
  masked value (`rhoCoins`, `Flock.Zk.rhoIndependent`), and level 0 closes only after all four;
- β = 0 in the inner proof: the Lean frame, at the inner proof's third round (`innerCoins`). The inner proof's four rounds
  leave through the rep's Lean frame, unmasked, in the form it fixes (`innerOwn`): `[ρ, σ]` and `Y` are the frame's own,
  from its samples and the coins, and of the other two rounds only `final_c'` and `τ`'s commitment are the ZK server's. The
  frame's `finish` comes after them.

Rust's own versions of the first two (`Masking::refuse_outer`, `mask_rank_ok`) still run under the ZK server, as
cross-checks that record instead of stopping (`Masking::defer`); the last two are also the Lean finish's, whose opening and
inner proof read the same coins after the frame. A rep any of them refuses where the Lean frame went on is refused
(`FIREWALL-REFUSED: Rust's masking refused the rep …`, `the Lean finish's opening refused the rep's coins …`) or, at the
inner proof, failed (`the Lean finish's inner proof refused the rep …`: its inner rounds the finish did not compute stop
the rep in the frame, §10.4), and the stop that leaves is always the Lean frame's. A device that claims degenerate coins (return
code 240) is a worker that failed, as above.

Each stops the session where those coins stand, which is where an honest device stops too. A table's proofs leave only
after both reps' Lean frames answer `ok` at their `finish`, and the proofs a prover without the barrier would send early
(`Plan::early_proofs`) exist only in a test harness's build.

### 10.2 The level-0 recheck

Before the session's `Commit` the ZK server recomputes the level-0 commitment itself, from the host witness
(`FC_HOST_WITNESS=1`: the ZK server assembles the witness on the host and uploads it whole, so the witness the recheck
commits is its own; the assembly is counted in the session's prove time, as the device's witness kernel is without it),
the table's level-0 hiding and its level-0 salts. The root of record is the Lean verifier's (`Flock.Firewall.Link`'s
commitment), the Lean session loop's, below. Rust commits
nothing of level 0 under the ZK server: the Lean finish keeps the table's tree for the reps' openings (§10.4). A root that is
not whole digests is refused before anything of the table leaves (`FIREWALL-REFUSED: level-0 recheck`). When the device's
root differs, the recheck's root leaves in the `Commit` and the session stops at its `Finish` (`FIREWALL-REFUSED: level-0
root`, §10.1). Its cost is reported per statement as `FIREWALL` lines (`t.firewall_recheck` wall seconds and
`cpu.firewall_recheck` `flock-circuit`'s own CPU seconds over the loop's level-0 request, which leave out the
Lean process's), beside the device's
`t.encoding_commitment`.

Through the Lean session loop (§10.4) level 0's hiding and the mask slot's words are the loop's, not the host's: the
prover sends the loop every table's packed witness in one request (`firewall::lean_level0`), and only once it holds them
all does the loop sample anything, so no table's witness can depend on any of the session's samples, as the analysis's
model fixes the witness before the server's randomness (`FlockSoundness.ZK.Table.view`). For each table it samples the
mask slot's words from the OS and writes them into the witness's mask slot (`Frames.Sessions.maskOf`,
`Frame.MaskSlot.fill`: the slot of the session's own statement for the table, at its `m`), samples the four extra lanes,
every lane's padding and the level-0 salts, commits level 0 on them and keeps the table's `Flock.Firewall.Level0` (its `m`,
level 0's schedule and `t_pad`, the witness, the samples, the codeword's rows, the Merkle layers from the leaves to the
cap, and the root, each `Link`'s function of the witness and samples: `Level0.ofDraws_ok`). The prover writes the words
into its own witness's mask slot (`z`, `a` and `b`, whose circuit is the identity there, and the lincheck's packing of
`z`) and takes the samples in place of its own (`Level0Zk`'s `extra` and `pads`, `LeafSalts::set_level0`), so the device
commits the codeword the loop did; the loop's root is the root of record. A worker that keeps words of its own in the slot
commits another codeword, and the session stops at its `Finish` (`zk_worker_level0`'s `own-mask`).

**The value commitments' salts are the ZK server's.** A statement's hm96 trees that commit a value (its public file's
`frame_v3` roots under a hiding commitment's schema, but a row's at known content, below) are committed by the ZK
server's value service (`core-firewall values --state DIR --salt-key FILE`,
`Flock.Firewall.Values` on the ZK server's record in `DIR`), which holds the server's 64-byte salt key `k` and derives
each leaf's salt from it: leaf `i` of
table `j` of the commit session `s` takes `y = SHA-512(k ‖ L ‖ s ‖ be32 j ‖ be64 i ‖ ctr)` for `ctr` 0, 1 and 2
(`Salts.saltOf`, `L` the SHA-512 of `verity/zk-server/value-salt/v1\0`). The serving stack sends each row's unsalted
64-byte digest and nothing else (`commit J D₀ N₀ S₀ …`, its body the digests, table-major; a body of any other length, a
salt beside each digest among them, is refused). Only once the request is whole does the service sample `s` from the OS, and
it answers each leaf's `b ‖ c`, each table's frame-v3-sha512 root and a tag: SHA-512 under `k` of the table's session,
index, domain id, leaf count, schema and root (`Salts.tagOf`), by which the server knows a table it committed. The prover
takes its witness's salts from the service with a refs file naming the tables (each `s ‖ be32 j ‖ D ‖ be64 N ‖ u8 schema ‖
root ‖ tag`) and the rows (each `be32 t ‖ be64 i`), every tag checked (`salts`, `Salts.saltsAnswer`), in place of the staged
file's (`flock-circuit --salt-refs REFS`, the service's address in `FC_FIREWALL_VALUES`).

The stack registers one commit session of the service's current epoch, the sessions it opened since its last
registration (`register S`, `Salts.Epoch`). The service numbers the registration on the ZK server's record, the next of
0, 1, 2, … with no gap, and appends its record there whole in the step that numbers it (a `register` event, under
`record.lock`: `Core.Firewall.register`). The record is `be64 n ‖ be64 m ‖ be32 T ‖` the session's `T` table refs `‖` a
record tag, SHA-512 under `k` of `n`, `m` and the table refs (`Salts.recordOf`), and the service answers it (`ok M LEN`,
then the record). It then opens a new epoch: a session registers once, and only in the epoch that opened it, and a
service started again opens a new one. A registration the record did not take is refused (`err the ZK server's record
took no registration: …`), its epoch closed all the same, and `flock-firewall values`, which keeps no record, refuses
every registration. `n` and `m` are public. `n` is protocol structure, and `m` is the number of sessions the epoch
opened: a stack that commits its values in `m` sessions and registers the one whose roots it likes leaks at most
`log₂ m` bits through the choice (`Verity.Guarantees.RegisteredLeakage`), and one that chose `m` from its values would
leak through `m` too. So the service of record fixes `m` at 1, protocol structure: an epoch opens one session, and a
commit while its session is not yet registered is refused (`err the current epoch's session is not yet registered: …`),
nothing of it answered. The stack chooses neither `m` nor a session, every record says `m = 1`, and the term is 0.

**Interior commits.** A two-stage audit's interior registration holds values that may follow the auditor's first-stage
sample, so the service commits it apart (`interior J D₀ N₀ S₀ …`, the commit request's form and body;
`Salts.interiorAnswer`), and nothing it answers of it depends on `k`. Once the request is whole it samples from the OS
`s`, each leaf's 192-byte salt and the session's own 64-byte key `ks`, and holds the salts and `ks` by session while it
runs, answering the salts to `salts` with an interior table's tag checked under its session's `ks`; a service started
again holds none. The tables' tags and the registration's record tag are under `ks`. At `register S` the service writes
`ks` on the registration's `register` event in the ZK server's record (`key`), with `ks`'s tag: SHA-512 under `k` of
`ks` and the record (`key_tag`, `Salts.keyTagOf`, its label the SHA-512 of
`verity/zk-server/interior-key-tag/v1\0`). The release side checks the record tag under `ks` only where `ks` is 64 bytes,
so the tagged message splits into key and record one way, and that tag is the server's
(`Salts.RegistrationRecord.valid`). So a writer of the record without `k` adds no interior registration the
loop admits, and a `register` event without a key is checked under `k`, which no interior record's tags are under. The
registration is numbered, appended and admitted as any other.

**Who reads an interior session's key.** `ks` and its tag are witness, never public. The tag is a MAC under `k`:
released, it would put bytes that depend on `k` in the auditor's view, which drawing `ks` from the OS keeps out of it.
`ks` is the key of the session's tags: whoever holds it can forge table refs the service takes as that session's. So
both live only on the ZK server's record, `record.jsonl` in its state directory. Only the ZK server's own OS user reads
that record: the value service and the session loop, and the tools the ZK server runs as that user
(`verity_flock.firewall`). `core-firewall values` makes the state directory that user's alone (mode 0700) before it
serves, and serves nothing where it can't. No process sends the record to the auditor (§10.5). Of an interior
registration the auditor may hold the registration's record, its roots and its tags under `ks`, and never `ks` or its
tag.

The release side admits each registration once, oldest first (the record's rule 5, `Core.Firewall.admitReads`, §10.5). Given the
key (`core-firewall session --salt-key FILE`; `FC_FIREWALL_SALT_KEY` for `flock-circuit --firewall-state`), every hm96
tree of a statement's public file must be a table of a registration on the record, at its domain, leaf count, schema and
root, whose record tag is the server's for its number, session count and table refs, under `k` or, for an interior
registration, under its `ks` once `ks`'s tag is the server's (`Salts.readsOf`), unless the verifier lists it as known
content (below); a statement that isn't is its recursive session's stop. The trees are held to the record once the
auditor has answered the session's `Register` and before the prover receives that answer (`Core.Firewall.seatReads`,
`Session.Hold`'s `known`): `Register`, one constant byte that carries nothing of the statement, has left by then, and
nothing else of a refused statement reaches the auditor. There a statement admits
each registration it reads that is not yet admitted, and passes over (`skip`) each older one it doesn't read: a
registration passed over is never admitted, a statement that reads one is a stop, and each skip is a failure on the
stream's budget (§10.5). A statement that reads only registrations already admitted admits none. `flock-firewall
session`, which holds no record, refuses `--salt-key`. Each leaf's `c` is the SHA-512 of its salt, so a witness passes
only on the server's salts: one on salts of the stack's own is rejected at its opening, where the value commitments'
claims are checked (`zk_value_salts`'s `kept`), and a statement the stack committed under its own salts is refused
once its `Register` is answered (`own`). A stack that registers more than once also chooses which registrations the auditor's statements
read: across a stream of `r` registrations with at most `b` passed over, at most
`log₂ Σ_{i≤b} C(r, i) + Σ_i log₂ m_i` bits, `m_i` registration `i`'s count, 1 on the service of record, so at most
`log₂ Σ_{i≤b} C(r, i)` (`Verity.Guarantees.RegisteredStreamLeakage`), counted in the stream's bound beside its stops
(§10.5).

**Known content.** A row the verifier registers itself at known content (`Flock.RegKnown`) commits no value and is none
of the service's. Its words and its salt are a function, fixed by the protocol under a tag of its own, of what is public
before the row is made, and the verifier computes that function itself and takes the row from no file of the stack's
(`Salts`'s header has the rule). The release side computes no known content: it takes a statement's known trees from
the verifier. After the draw, the verifier's answer to the session's `Register` lists each tree of the statement that it
computed itself: `be32 K`, then `K` entries of a tree's 64-byte domain, `be64` row count, schema byte and 64-byte root
(`flock-verify known-trees` writes the list from the verifier's own `--registered` file and the statement's public
file's header, `Registered.knownTrees`; `flock-circuit serve --known-trees LIST` sends it; an answer that ends at the
draw lists none). The loop cuts the list off the answer, so the prover receives the draw alone, and holds the
statement's trees to it before `Hello` leaves (`Salts.readsKnown`): a tree that equals an entry reads no registration
and is recorded (a `known` event, no failure), and every other tree must be a registration's table as above. A list that does not
parse, an entry whose root is a registration's table's, two entries of one domain, an entry at whose domain the
statement has no tree, and a tree of the statement that differs from the entry at its domain are each the session's
stop. So a session that passes holds exactly the listed trees as known content, and the stack chooses neither which
trees pass so nor a bit of them. The list is the verifier's word only where the answer to `Register` is the auditor's:
nothing of the stack's reaches the auditor but through a release side run with the key.

Three limits. Without `--salt-key` the loop checks no statement's value commitments and admits no registration: nothing
yet requires the key of a deployment's loop. The ZK server numbers one sequence of registrations for all its
streams: a skip counts on the stream of the statement that passed it over. And the epoch is the service's process's: a
service started again opens a new epoch, in which the session of the one it cut short never registers, so each restart
of the service mid-epoch, the ZK server's own act (`zk-server-own-user`), gives a stack one more session for that
registration.

The tags, the record tags and the count mean what they say only while nothing the stack runs reads the key, the epoch or
the record: the key file is created its owner's alone, the state directory is made so, and the service and the loop run
under the ZK server's own OS user, which no worker and no serving stack shares (`zk-server-own-user`,
`Verity.Assumptions.ZkServerOwnUser`; the selftest runs them all under one).

### 10.3 The rule, inner phase (`rec_live`)

| item | rule | why the circuit prover has no free choice |
|---|---|---|
| `Hello` | constant | the ZK server compares the prover's, Σ aside, with the pinned schedule's as canonical JSON (`rec_live.canonical`: sorted keys, no spaces, so `2` and `2.0` differ) and forwards the schedule's own, never the prover's text or object |
| `Commit`'s level-0 roots | salted | each 64-byte node of each root reaches the challenger as a registered row's `b ‖ c` (`rec_live.top_rows`: its 32 u16 words, then 32 zero words; `hm96-sha512/row/v1`) under its own fresh 192-byte salt; the roots and salts stay in `private.json` |
| `Commit`'s tables and their order | constant | the pinned schedule's `roots`, each with its node count, in the schedule's order (their names, sorted: `Flock.Firewall.Schedule.tables`); the ZK server refuses any other order, so the order carries nothing of the prover's |
| a round's caps (its `BYTES` frames that commit a tree) | salted, or counted | a fresh top as the roots, one fresh salt per node; a copy of a level-0 top committed at `Commit` (the rep's binding round's cap, which must be `root_B`, and Ligerito's first absorb) is counted (`level0`), never committed again. Copies are by position, never by value: the round's first `level0` caps, as the schedule names them, must each be its table's top committed at `Commit`, and every other cap is a fresh top, whatever its value (`Flock.Firewall.copiesOf`) |
| a round's free message | salted | `c_k`, its one row committed under a fresh salt (as before) |
| `Register` | constant | the ZK server answers it itself with the verifier's sample, every unit of the population (`rec_live --draw FILE`, `rec_live.full_draw`: the stratified sampling design at `K = N`, so each stratum's `k` is its size), and the record holds that sample as `unit_draw` from the session's start; `verify --stratified N --program --partition` derives the same sample and refuses any other |
| statement digest, labels, domain, nonces | dropped | public constants or functions of the hidden statement; never forwarded |
| `Link`'s `y` | constant `0` | the schedule fixes its length (`link.y` elements of 16 bytes); the ZK server refuses any other byte length, and any nonzero byte |
| the order of `Hello`, `Commit`, `Link` and `Finish` | constant | the ZK server itself refuses a `Commit` before `Hello` or after the first, a `Link` before `Commit` or after the first, and a `Finish` before the schedule's last round (counted from its own streams, never by reading the challenger), each before any call on the challenger: a request out of its place reaches the challenger as a plain stop at the same point |
| stream names, `Open`'s parent and position, each round's coin count, its fresh tops' node counts (`tops`), its copies of a level-0 top (`level0`) and its message rows (`commits`: one, or none for a round without a free message) | constant | the pinned schedule's; the ZK server refuses any other. The challenger's record lists the streams by name, as the schedule does, never in the order the prover opened them. An `Open` may come any time before its stream's first round, so the ZK server keeps it and opens the stream at the challenger just before that round: neither the challenger's calls nor its record show whether or when the `Open` came, and a stream is listed from its first round |
| how the session ends | `finish` or `stop` | every session served is written and counted toward `--sessions`, with `end` `finish` or `stop` (a refusal, or a session that ends without `Finish`: closed, reset or cut inside a frame), never why |
| a request the ZK server cannot read, or the ZK server's own fault | refusal | any exception while handling a request stops the session (`FIREWALL-REFUSED: a request the firewall cannot read (…)`), never the ZK server |

The schedule is `rec_live --schedule FILE`, and the proxy refuses to serve without one. `rec_live --schedule-for STATEMENT`
derives it before any session from `flock-circuit statement`'s record of the public file alone (`schedule_for`: `dense_m`,
`k_log`, `nbl` and `regions`, the four numbers the schedule reveals anyway), and `serve` writes it to `schedule.json` before
the first session. No session pins it: a schedule taken from a session the prover ran would be the prover's choice
(`--schedule-of PUBLIC_JSON` prints a record's schedule, to compare with the one it was held to). The proxy and
`flock-firewall` refuse a schedule no honest session can finish before any request (`check_schedule`,
`Flock.Firewall.Schedule.canonical`): its `Hello` must print back as it parses (naturals, and strings without `"`, `\` or
a control character) and name no `coins`, each round must be its six fields and nothing more, and each stream's rounds,
at least one, must come in the schedule's order numbered 0, 1, … as many as its `rounds`. `flock-firewall`, which holds
no registered top, also refuses a stream, or a copy of level 0's top, whose table has no root. The record says how its tops
are salted (`"tops": "hm96-sha512/v1/node-row"`, `rec_live.TOPS`). A refusal (`FIREWALL-REFUSED: …`) stops the session: every later
request is refused with the same reason, `Finish` answers not accepted, and only `public.json` (what the challenger had
received, `end: "stop"`) is written.

A salted node's opening (node and salt) is the prover's. V*'s inputs (`rec_vstar.tops`) read the clear tops from
`private.json` and refuse `FIREWALL-REFUSED: tree top` when one does not open its commit string in `public.json`, or when the
record's tops are not salted. The outer circuit climbs to the committed tops: `rec_outer` registers each level's tops
(`TopValue`) under the ZK server's very salts, and the verifier builds its entry from the recorded `b ‖ c` alone. A node is
committed as a registered row is, so climbing costs `RecOpen` no hashing of its own: the verifier
builds a registered value from the challenger's `b ‖ c` (one row a node, under the ZK server's salts), and each opening's
`out` reads the row of the node it climbs to. The ZK server pays one SHA-512 a node more than for a bare node (the row
digest).

The rule is also a Lean program, `Flock.Firewall.run` (`../verifier/Flock/Firewall.lean`, the executable
`flock-firewall`): the pinned schedule, the worker's requests and the ZK server's salts in, the challenger's record out. Its
message type is the contract: each message is the schedule's own value, a count it fixes, zeros, or commitments under the
ZK server's salts, so nothing a worker sends reaches the challenger any other way. `tests/test_firewall_agreement.py` holds
`rec_live`'s proxy to it byte for byte (the record without the coins it flipped), on an honest session and on every prefix,
drop, duplicate, swap, forgery and seeded byte flip of one. One difference is deliberate: Lean's JSON keeps no number's
spelling, so `flock-firewall` refuses any `Hello` with a fraction or an exponent, where the proxy compares Python's canonical
JSON. The two agree whenever the schedule's `Hello` holds only integers, as `schedule_for`'s does.

The send check is that program itself. `rec_live --lean BIN` (required) starts one `flock-firewall live` a session (`Flock.Firewall.liveWith`): its first line is the schedule, and
each request goes to it before the proxy handles it. It answers with the messages the challenger's live-coin part gets for the request and
the salts it generated for them (`Flock.Firewall.liveStep`, one step of `run`'s loop, its salts from the operating system, those
a request doesn't sample kept for the next), or with the refusal that stops the session. The proxy delivers Lean's messages, in Lean's order, and handles the request too,
for the prover's side (the answers, the session record, `private.json`) on Lean's salts: a message or a salt of its own that
is not Lean's, or one of Lean's it leaves unused, stops the session. The same test holds the relay to `run` at the salts the
live session generated, on every case and every whole placement of the `Open`s. On the challenger's coin tree (`--coin-tree`)
the first line also gives the tree's spec: the prover's `Hello` names the tree, with a nonce of its own the session never
uses, and `Flock.Firewall.treeHello` holds `coins` without its nonce to the spec and hands `step` the `Hello` without
`coins`, so the challenger's live-coin part gets the live-coin session's messages; the same test holds the proxy, the relay and `run` to
one another on the tree's sessions. Lean's ZK server refuses a binding round of a registered top, so `rec_session.serve`'s
sessions, with their stacked table's registered top, keep the Python check that the test holds to `run`.

In the Lean analysis (`security/Proofs/Flock/Recursive/`), `FirewallTemplate.run_blank` proves that a session `run` finishes
sent the pinned schedule's transcript (`tmpl`) but for its commitment strings, and `sends_of_one` that if `run` finishes at
one salt list with each commitment string the commitment of a given 64-byte digest under its salt, then at every salt list
of that length it finishes and sends the transcript with each string the same digest's commitment under the new salt.
`FirewallAccepts` states the honest worker's half: at the zero salts, `run` finishes on the honest requests, with the tables
named and each commitment string that of the honest message's digest. `firewallSends_of_accepts` derives `FirewallSends`
(the hypothesis of the recursive ZK statement, `RecursiveFirewallPrivate`) from it at the transcript `firewallT`.
`FirewallLive.run_live` proves that a live session that doesn't stop (`liveRun`, `liveWith`'s loop over `liveStep`) sends
what `run` sends on its requests at the salts it read, in order, and `run_append` that a finished run is the same at more
salts after its own; so `liveSends` (and `liveSends_of_accepts`, from `FirewallAccepts`): on the honest requests, a live
session that doesn't stop sends the transcript at the commitment strings of the first salts it read, and finishes.

### 10.4 The prototype (`FC_FIREWALL=1`)

`flock-circuit prove --zk` with `FC_FIREWALL=1` runs the ZK server (`firewall.rs`, and `Firewall` in `flock-circuit.rs`) on the GPU path and
the CPU path alike, for a statement of one table on the device and of one or several (`--session-tables`) on the CPU; it
refuses to start without `FC_HOST_WITNESS=1`. It is stricter than
§10.1 in one place: the session loop recomputes the whole opening, level 0's lane rounds included, in lock step with the
device (`Drive.openStep`, below). Each of the device's opening rounds must equal the loop's byte for byte, and the opening
that leaves is the loop's. The lincheck's folded comb is the ZK server's (the Lean finish folds it), and `final_c'` is
computed from the passed values. The algebraic phase's rounds are held to the statement's scheduled rounds (domain,
labels, each item's op, kind and length, the coin counts). At the end the returned proof must have the statement's shape (each
zerocheck and lincheck vector's length), the replay of the values the ZK server passed must give the proof's exposed values, and
the zerocheck and lincheck that leave are rebuilt from the passed values (`zk_veil::from_exposed`), as the opening that leaves
is the loop's: nothing of the device's returned proof leaves.

Every round that leaves is computed in Lean, from randomness Lean samples, and the coins' refusals are Lean's. Each rep's
frame (`Flock.Firewall.Frame` in the Lean verifier) runs in the session loop below (`FC_FIREWALL_LEAN` names the
`flock-firewall` binary, by default the Lean verifier's build in the checkout, and a session behind the ZK server refuses to start
without it), and the prover runs one `flock-firewall tap` for its other samples (§2), started before its first sample.

The session's `Hello` is Lean's too: the loop's, on the statement's files (`--statement`, `--circuit`, `--public`,
`--tables`, `--session-tables`) at the auditor's sample: the session's `SessionConfig` as canonical JSON with a 32-byte coin
nonce the loop samples from the OS, the Lean `CoinLayer.helloJson`. A setup it refuses stops the session before it sends
anything. `flock-firewall hello` computes the same `Hello` on its own (`test_lean_hello.py`): each `draw LEN` and the
auditor's sample (or `null`) is answered `ok N` and the `Hello`, or `refused WHY`, and `draw LEN NONCE` fixes the nonce.
`FC_FIREWALL_STATEMENT` names the Lean statement (by default the build's, `+seed-injection` under that feature), and `FC_FIREWALL_HELLO_ARGS` adds arguments:
`--partition F --program F` for a statement whose setup reads them, and the verifier's own sampling-design inputs as
`flock-verify verify` takes them, which a stratified, work or closure sample sets up only on (U2, U3): `--stratified K`,
`--work K --work-table F`, `--closure F`.

The session itself goes through Lean: one `flock-firewall session` process (`Flock.Firewall.Session`,
`firewall::LeanSession`) per verifier address and table count, on the same statement arguments, listens on loopback and
holds the auditor's connection for every session the prover opens to it. A `prove --stream` process, whose sessions are of
different statements, starts its tap before its first sample, as `prove` does, and one loop for each session instead, on
the circuit and the tables the session's line names: the loop starts at the session's turn, once the session before it
has its verdict, and ends once its own verdict is emitted, so the stream proves one session at a time (it refuses
`FC_VERIFY_AHEAD` above 0, and no session's front runs ahead). The statement's name and the `Hello` flags are the
process's for every session, so one stream serves sessions of one partition and one program. The stream itself does not
check that: a line has no field for them. A session whose circuit binds another partition or program is refused by its own
Lean loop when its statement is set up (`HmRow.checkPartition`). A line the reader refuses ends the process at once, under
the session then being proved, which gets no verdict. A session left without a verdict is failed and never proved again,
since a second session for it has new coins. No byte the prover sends reaches the auditor:
every frame that leaves is the loop's own encoding of what it decided (`Session.roundFrame`, `Session.commitFrame`,
`Session.State.wire`). `Register` and `Finish` leave as their tags, a `Link` as zeros at the session's link points, an
`Open` as its rep stream's at the stream's domain, a `Round` as the schedule's stream and count around the frame's release
(bytes after its count dropped), and a `Commit` as the session's Σ, its tables' level-0 roots and no public words. A
`Ping` never leaves: the loop answers it `Ok` itself, as `rec_live` does. A `Proof` the prover sends is answered `Ok` and
dropped: the proofs that leave are the loop's finish's files. Each request leaves at most once, at its place in the honest
prover's order (`Session.State.place`): `Register`, `Hello`, the first rep stream's `Open`, `Commit`, `Link`, that
stream's rounds, then each later rep stream's `Open`, once the rounds before it are past, and its rounds, and `Finish`
once every rep stream opened. A second `Register`, `Hello`, `Link` or `Open` of a stream, any of them out of its place, a
`Prime`, a frame of a tag it doesn't know and an `Open` of a stream the session doesn't have are refused. So what leaves
is fixed by the statement and the schedule, as the model has it: each frame that carries no coin once, at its place, and
one message a round (`Claims.ZKServer.LeanRelease.toSrv`), and nothing the prover places. It sends its own
`Hello` (the Lean `Hello` at the auditor's sample and a nonce it samples from the OS) in place of the prover's, checks
`Hello`'s answer (20 words, key bit 2047 clear) and every coin opening against the root, and hands the prover only the
coins, as `coin_tree::Checked` does; an answer to a `Commit` or `Round` that is not coins stops the session, refused
(`Session.State.noCoins`), so no opening is asked for twice. `Hello`'s answer itself never reaches the prover: once it
checked, the prover receives in its place the loop's id for the session, four words sampled from the OS with the nonce
(`Session.idFrame`). The tree, link count and nonce it checks them under are the values it set up and
sent (`Session.helloOf`), never a parse of the `Hello`: the setup's coin tree, whose streams must be the session stream
then each table's `rep0` and `rep1`, distinct (`Session.streamsOk`), else `Hello`'s answer is refused. It asks for each
opening at the schedule's place, not the prover's: the session stream's at `Commit`, then every round of each rep
stream in table order, rep 0's before rep 1's, at the coin counts the verifier's replay samples (`Session.planOf`;
`flock-firewall session --schedule` prints them). A `Commit` or `Round` anywhere else is refused before it reaches the
auditor, and so is a `Finish` before the schedule's last opening. A request it refuses is answered
`Refused("PROVER-STOPPED: …")` and stops the session: every request after it is refused for the same reason without
reaching the auditor, a `Ping` alone answered `Ok` by the loop, so nothing of the session reaches the auditor after its
stop, and the auditor's connection closes at the session's end as at any other stop. The session ends at its `Finish`,
whatever the loop decided of it: once the prover has its verdict or refusal the loop reads nothing more of it and closes
both connections, so no `Finish` and no proof leaves twice. The reps' frames are the loop's own, and the loop drives
them (`Frames`): it prints `FRAMES 127.0.0.1:PORT` after its `LISTENING` line, and a connection there whose first
line is `rep SESSION I R` starts table `I`'s rep `R`'s frame for the live session named `SESSION` (`Frames.startRep`,
`firewall::LoopRep::start`), at the statement's start (`Drive.start`), once the session's level 0 is sampled (a rep
before it is refused, so the pads' root, too, reaches the prover only after every table's witness is in), and is
answered `ok 64` and the pads' root alone:
the samples the loop made for the rep (the pads `h`, `μ`, their padding, `τ`'s salt and the pads tree's salts) never
leave it. Its frame masks the rep's rounds with them, and its finish proves the rep with them (`Finish.drawsOf`). Every
request the frame takes is the loop's,
on the schedule `Drive.reqs` fixes: a rep stream's `Round` carries the device's round, and the loop places it as the
algebraic phase's scheduled round (`Skeleton.rounds` on the table's level 0), then the opening's (`Drive.openStep`, the
rep's opening a round at a time in the loop, on the extra claims at the `Commit`'s coins and the rep's salts for levels
≥ 1, one copy with the finish's), then the inner proof's, each at the frame's own round (`Frame.innerSched`) on the loop's
finish's round, which the finish hands the loop before Rust relays it (§10.4). At an inner round the finish has not
computed, the loop hands the frame no device round (`Frames.roundStep`): whatever Rust relays fails, the frame's own round
stands in, and the rep stops. Once any rep's frame of the session stopped, the loop hands no rep's frame of it a device
round (`Frames.Sessions.stopped`), so every rep of the session stops at its first failure, each round leaving as the
statement's. Rep 1's frame first takes rep 0's algebraic-phase coins. What leaves is the
frame's release, at the stream's first round after its domain frame (`Frames.round`); anything the frame refuses is
refused before it reaches the auditor. Once a round's coins open the tree, they are the frame's coins before the prover
receives them (`Frames.feedOf`, `Frames.coins`), and the session's `Finish` leaves only once every rep's frame answered
`finish` not stopped (`Frames.finish`). The prover sends nothing on the frame: on a connection there it asks a rep's next
place (`place SESSION I R`: `alg N`, `open N` or `own`, by which it ends a failed rep's rounds) and, at the rep's end, the
values the frame took (`vals SESSION I R`, `ok N` and the words in absorbed order, for its end check), and it passes every
device round through, unmasked, to the loop (`firewall::Gate`).
A connection there whose first line is `level0 SESSION M₀ LEN₀ … M_{J−1} LEN_{J−1}`, a pair for each of the session's
`J` tables in table order, followed by each table's `LEN_j`-byte packed witness in that order, is the session's level 0
(`Session.level0`, §10.2) for the live session named `SESSION`: the hex of the loop's id for it, the four words the
prover received for its `Hello`'s answer (`Level0.nameOf`, `firewall::level0_session`). The loop reads every witness
before it samples anything, then answers each table in order `ok N R` and the `N` bytes of the mask slot's words, the
extra lanes, the paddings and the salts the loop sampled, then its `R`-byte root, or the request `err WHY`, nothing sampled
kept; the loop wrote the words into each witness's mask slot before it committed level 0, whatever the witness held
there. A session the loop isn't serving is refused, and so is a request whose witnesses are not one for each of the
session's tables, an `M` other than the table's statement's, a witness of other than that `M`'s words, or any request
after the session's level 0 is sampled, a witness after a sample among them: the loop samples a session's level 0 once, on
every table's witness (`Level0.Slots.draw`), so the prover can't sample again until a root suits it, nor hand a witness
that reads a sample. The selftest's `zk_worker_level0` holds a worker to that beside an honest session:
`FC_LEVEL0_TAMPER=second` asks again, every table's witness once the loop answered, and `past` asks first with one
witness more than the session has tables; the loop refuses each, and the session goes on on its one sample and is
accepted. The
session's `Commit` leaves only when its roots are its tables', in order, root `i` the one the loop generated for that
session's table `i` at its statement's `m` (`Level0.commit`; a root sampled at a smaller `m` would pad too few positions).
Every `Level0` sampled for a session, committed or not, is freed when it ends (`Level0.Slots.close`).
The proofs leave just before `Finish`, the loop's finish's file of each table's rep 0 and rep 1 on the level 0 the
`Commit` bound (`Session.ownProofs`, kept once a rep, `Frames.Sessions.file`), every rep's or none, and only once every
rep's frame ended not stopped; where one stopped, `Finish`
leaves without them once the schedule's last opening has, and the prover's `Finish` is refused (`PROVER-STOPPED: the Lean
frame stopped table I rep R`). The analysis's release is this loop
(`Claims.ZKServer.LeanRelease.loop`, `Flock.zk_server_lean_loop`): its `Hello`, requests, last request and the coins its
frames take are proved (`loop_coinsRead`). At the loop's own frames, link and `Finish` (`LeanRelease.ofLoopLink`, the
release `Flock.SecurityProofs.SessionLoopZKDeployed` is about) the frames' state is proved (`ProgReads`, `ofLoop_progReads`:
the `prior` coins and the rounds Rust hands them), and so, in the masked verifier's scope where the loop's inputs from its
host are the tape's (`Loop.Src`), are its link (`LinkRoundEq`, `linkRoundEq_ofLoopLink`) and the `Finish`'s record and
proof files (`FinishIs`, `Loop.finishIs`). The frames refusing every coin fault is proved at the loop's own frames
(`frameUnrefused_ofLoop`), and from it, `FinishIs` and the tables admitting unrefused coins, proved at the deployed
tables, `FinalRoundEq` follows (`finalRoundEq_ofLoop`). The other bytes it releases the theorem still reads only through
named hypotheses: the worker's passed rounds and the stand-in rounds are the session's (`LiveRoundEq`, `FillRoundEq`) and
the inner proof's round 0 is the ZK server's own (`LoopInnerRound0`), and of the tables' side the switched round
encoding causal at the coins the loop computes at (`SwCausalOn` at `Loop.SwOk`: no zerocheck outer coin `1`, which the
frames refuse) is a premise; the rest of it (`Tables`, `Complete`, `DummyReg`, `Causal`, `TapeOK`) is proved at the
deployed tables (`Loop.tables_dataE` and its siblings). Until those are proved, the ZK server is not yet proved
zero-knowledge.

The frame's protocol, which the loop steps on its own requests (`Frame.Req`) and `flock-firewall frame` serves on its
standard input (`tests/test_firewall_frame.py`):
- **start**: one JSON line with the rep's layout (`n_mlv`, `n_lc`, `n_ood0`, `k0`) and the statement's mask slot (`w0`,
  `n`, `blocks`), nothing else. The frame samples the rep's randomness from the OS (`Draws`: `h`, `mu`, `pad_h`, `pad_mu`,
  `τ`'s salt, the pads tree's salts) and answers `ok 64` and the root of the pads tree it computed over them
  (`Frame.padsRoot`); the samples never leave the frame. `flock-frame-tape`, a test's executable, runs the same frame on
  samples read from a file, so `tests/test_firewall_frame.py` knows the pads a frame masks with.
- **prior** (rep 1 only, before its first round): `prior N` and the table's rep 0's `N` algebraic-phase coins, whose claim
  points the rank check covers too.
- **round**: each device round, its frames with their values unmasked, then the statement's round in its place (`round N LEN REFLEN`): the frame's pass decision (`check`, §10.1), answered `ok LEN' pass`
  and the device round's release, or `ok LEN' fail WHY` and the scheduled round's, the rep then stopped. A round leaves
  with the pads' label and root after the zerocheck's label, each value absorbed in a masked phase as `v + h_i` under the
  next pad of `Layout::absorbed`'s order, everything else as framed, and no domain frame. A round marked `fail`
  (`round N LEN fail`, the round in its place alone) leaves as that round, and a round the frame refuses itself releases
  nothing: each stops the rep, whose every later algebraic value leaves as its pad alone
  and whose `finish` is `stop`. At the opening's label the frame runs the rank check, and a mask slot of less than full
  rank is answered `stop WHY`.
- **coins**: after each released round, `coins N` and the coins the stream answered it. No round and no `finish` is taken
  while they are owed. In the session loop they are the coins the loop opened for the round (`Frames.coins`). A zerocheck outer coin 1, level 0's `ρ_0, ρ_1` dependent over F128 (the four words after its last
  masked value) and the inner proof's `β = 0` are answered `stop WHY`.
- **the inner proof**: its four rounds (`ε`, the batching coins, `β`, the queries), after the opening's, are the
  ZK server's own and come alone (`round N LEN`), which the frame takes nowhere else: the Lean finish proves it
  (`inner J R LEN`, below), and each of its rounds goes through the rep's frame before it leaves, and the frame takes one inner proof, after the opening
  ended, and no round after its fourth. It takes each round only in the form it fixes (`innerOwn`): the inner label,
  `final_c'` and `ε`'s squeeze; `[ρ, σ] = [ε h_fa + h_tx, h_fb + h_ty]` and the batching coins; `τ`'s 64-byte commitment
  and `β`'s squeeze; `Y = (h ‖ pad_h) + β (μ ‖ pad_μ)` and the queries, with `ρ`, `σ` and `Y` its own values from its samples
  and the coins. Any other round (a frame more, a value changed) is answered `err WHY` and releases nothing. The session
  loop sends each with the frame's own round in its place (`round N LEN REFLEN`, `Frame.innerSched`: `innerOwn`'s form at
  the finish's `final_c'` and `τ`'s commitment), so a relayed round that differs stops the rep, its own round leaving.
  At a round the finish has not computed, the loop sends no device round (`LEN` 0) whatever Rust relayed, and the
  frame's own round, zero words in place of the finish's, leaves as a stopped rep's.
- **finish**: `ok`, or `stop` for a stopped rep, once every masked phase closed with every pad of the layout used and
  the inner proof's four rounds left.

A `stop` ends the rep where the coins stand, as an honest prover's refusal does. The prover's masking challenger still
runs on the device's rounds: its inner challenger (`firewall::Gate`) drops what it would send and hands the stream the
device's round unmasked, which the rep's frame checks and masks; the inner proof's rounds go through it the same way. The
recheck, the opening, `final_c'`, `τ`'s commitment and the inner proof's openings are Lean's (the loop's and its finish's),
and the inner proof's `[ρ, σ]` and `Y` the frame's (the values the ZK server fixes, `FirewallComputes`). Each rep's rank
check logs `FRAMERANK` (the frame's) beside `ZKRANK` (Rust's reference, over Rust's claim points).

**The proof file is the Lean finish's.** Each session has one Lean finish (`Flock.Firewall.Finish`), the loop's own (a
connection to its frames port, `finish LEN` and the sample Register answered, `Finish.serveLoop`), at the statement the
loop's `Hello` is set up from. Each table's recheck hands it the table's root (`level0 J LEN`), and the finish takes the
loop's `Level0` of that root (§10.2), its rows, layers and cap, without committing again; the root it answers must be the
root of record. `flock-firewall finish` serves the same protocol as a process of its own, from the statement's files and a
table's level-0 request (`table J LEN`: the witness, extra lanes, paddings and level-0 salts), which
`../agreement/firewall_finish_agree.py` drives. With the table it samples each rep's salts for levels ≥ 1
itself, from the OS (`salts J R 0`, `192·2^d` bytes a level), and hands them to the prover, whose device trees at those
levels take them (`LeafSalts::preset`), and so do the loop's opening rounds: no salt of the rep's proof file comes from the
prover. The loop's finish samples them only once the rep's frame took the coins of its algebraic phase's last round
(`Finish.Own.algebraEnded`): the loop then answers the rep's `place` with `salts` until they are sampled, and the ZK server
asks for them at the round that opens the opening, before the device builds its first tree past level 0. A `salts`
request before then, on any finish the prover opens, is refused, so no round of the algebraic phase can depend on them,
and so is one after they were sampled: the loop keeps a rep's salts once, whichever of its finishes generated them, so the
prover never holds two samples of them.
No request carries the rep's samples: the loop's finish proves each rep with the ones its frame took
(`Finish.drawsOf`, from `Frames.ownOf`), and refuses a rep no frame holds. After the rep's end check the ZK server sends
the finish the rep's request with the coins of its PIOP and opening (`inner J R LEN`: the values the frame took, the
extra claims' points and the coins), and the finish answers the constraints it replayed (`cons LEN`, which the ZK server
holds to the statement's relations, `relations_held`), then each round of its inner proof, which leaves through the
rep's frame, and last the names of the constraints that fail on the rep's pads (`Finish.namesBytes`, the bucket's
`constraints_failing_on_pads`). A constraint's value on the pads is its check on the unmasked values and the coins (each
masked value's pad cancels), so the names tell the prover nothing of the pads. Where the finish refuses the request (or
the ZK server its constraints), the ZK server fails the rep and sends each inner round the finish did not as no device
round, its squeeze alone at the round's coin count (`ε`, a batching coin a constraint, `β`, the queries): the frame
fails each and its own round stands in, so the rep stops in the frame and the session stops at its `Finish`, every round
on schedule. The finish ends at a refusal, so a later rep's requests find none. Then, where none fails, the finish gets
the rep's request again with every round's coins (`rep J R LEN`) and computes the rep's whole proof file from it, the
rep's samples and its own salts: where that request's inputs and coins are the ones the `inner` request took, the file
encodes the proof that request computed (`Finish.Kept`), which is the same proof. The loop's finish serves only that
file: it refuses a `rep` request no `inner` request answers, and a `table` request (the host's level-0 data), so every
file it serves was computed live on the loop's level 0 (`Finish.serve`'s `liveOnly`). `flock-firewall finish` alone, the
offline cross-check, samples each rep's samples itself at its `salts` request and still computes a file from the `rep`
request's inputs. That file is the one that leaves; Rust
computes no opening, inner proof or proof file of a rep behind the ZK server. A rep that a check failed, or whose frame stopped
it, leaves no proof file, so the finish never gets its `rep` request. The bucket's `t.lean_finish` is the request's wall
seconds, and `t.zk_inner` the inner proof's.

**What `flock-circuit` still computes** under `FC_FIREWALL=1` is the prover workers' side: the host witness
(`FC_HOST_WITNESS=1`), and the device's (or the CPU prover's) rounds and returned proof, which the Lean frame passes or
fails against the loop's. Its own checks (the device's level-0 root against the recheck's, the end check on the returned
proof against the values the frame took, `relations_held` on the constraints the finish replays) can only refuse more. It moves the rest: the
rounds and coins between the device, the frames and the session loop, and the finish's proof files as the session's
`Proof`s. Under the ZK server Rust's level-0 commitment (`commit_zk`) and opening are the worker's values only, held
to the Lean's and never released, and Rust computes no proof file (`zk_finish`). As what leaves, they remain only on the
unserved `--zk` path, which no ZK server serves.

**What the worker sees.** `flock-circuit`, the device and the host process around it alike, never receives a rep's
pads: the rep's frame answers its start with the pads' root alone, the masking challenger runs on zero pads and hands the
frame each round unmasked, the frame masks what leaves, and the finish proves the rep on the frame's samples and answers
the prover only the names of the constraints that fail on them. A worker that sets a masked value to a word of its own
choosing finds it leave as that word plus a pad it never saw, and the session stop at its `Finish` (`zk_worker_pads`,
below). What the worker still sees is level 0's hiding: every lane's padding, the four extra lanes and the level-0 salts the
loop sampled for each table (§10.2), because the device's level-0 commitment needs them, beside each rep's salts for
levels ≥ 1, which its trees take and which it gets only once the rep's algebraic phase ended. None of them leaves the
worker a free choice: the ZK server rechecks the device's
level-0 root against its own commitment of the host witness under that hiding and those salts (§10.2), and every byte of
the opening, level 0's values included, must equal the loop's own opening round (`Drive.openStep`) byte for byte, so
nothing of level 0 the worker sends leaves except as the ZK server computed it. The worker also sees the mask slot's
words, which the loop samples and writes into the witness it commits (§10.2), and which leave the worker no free choice
either: a worker that keeps words of its own there commits another codeword, and the session stops at its `Finish`
(`zk_worker_level0`). Of the verifier's messages it sees only the coins, stripped of their openings: never `Hello`'s
answer, the coin tree's root and the session key, in whose place it receives the loop's id for the session, four words
the loop samples, which it names the session by on the loop's frames port.

In a session of several tables each table has a ZK server of its own in turn: its salt context and recheck, and its reps'
frames in the loop, one table's ending before the next table's begins. A later table's level-0 root leaves in table 0's `Commit`, before any
stream opens, and it is the Lean root of the host witness under the table's level-0 hiding and salts (the loop's, else the
Lean finish's of the table's level-0 request, the recheck's own computation), so no root the device computed is in that
`Commit`. Before anything of a later table is sent,
its recheck must give the root that `Commit` carried (`FIREWALL-REFUSED: circuit-01's level-0 root: the session's Commit
carried … where the recheck has …`), and the table's device root is then held to it like table 0's.

`FC_FIREWALL_TAMPER` makes the device dishonest in one named way. Each is refused by name, and none stops the session
before its `Finish` (§10.1): the tampered message never leaves (a masked value leaves as `v + h`, uniform whatever `v` is),
every stream runs its schedule, and `Finish` goes without proofs, the auditor's verdict not accepted after as many rounds
as an honest session's. The prover's record has the refusal: `prover_stopped` the ZK server's, where its own checks caught
it, and the verdict's `why` the loop's `PROVER-STOPPED: the Lean frame stopped table I rep R` where the rep's frame stopped
it (both, for some). `../agreement/firewall_controls.sh` runs each control through the loop and holds it to that outcome,
after an honest session that must be accepted with no rep's frame stopped and no `prover_stopped`. For `opening`,
`opening-frame`, `cut-tree` and the inner rounds (`FC_INNER_TAMPER`, §10.4) it requires the frame's stop and admits, never
requires, the ZK server's
`FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` beside it: the session stops at its
first failure, every rep with it (§10.1), so the ZK server's own check of the proof against the values the frame passed may
refuse it too. `later-NAME` is `NAME`
on every table of a several-table session but the first, and is refused
the same way at the second table:

| `FC_FIREWALL_TAMPER` | the device | refusal, at `Finish` |
|---|---|---|
| `root` | flips a bit of its level-0 root | `FIREWALL-REFUSED: level-0 root: the device committed … where the recheck has …` |
| `witness` | proves a witness one bit off the host's | `FIREWALL-REFUSED: level-0 root: …` |
| `label` | flips a label byte in the algebraic phase | the frame's stop at the round; `FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` |
| `opening` | flips an observed value in the opening's first round | the frame's stop at the round; the ZK server's `FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` may come with it |
| `opening-frame` | flips an observed value in the opening's second round | as `opening` |
| `algebra` | flips a masked value, in its frame and its proof alike | `FIREWALL-REFUSED: circuit: the constraints … fail on a rep's pads` |
| `steer` | sets each rep's first masked value to a word of its own, in its frame and its proof alike | as `algebra`; the word reaches the verifier in no request |
| `algebra-proof` | flips a masked value in its returned proof only | `FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` |
| `shape` | appends a value to its returned proof's `round1_ab`, past the ones the ZK server compares | `FIREWALL-REFUSED: the proof's shape (…) is [65, 64, …] where the statement's is [64, 64, …]` |
| `cut-rc` | returns an error at its round `FC_FIREWALL_CUT_AT` | `FIREWALL-REFUSED: the device failed: CUDA circuit prover returned error 1` |
| `cut-rc240` | claims degenerate coins (return code 240) at that round, on coins that are not | `FIREWALL-REFUSED: the device failed: PROVER-STOPPED: degenerate coins: …` |
| `cut-panic` | panics at that round | `FIREWALL-REFUSED: the device failed: panicked: the worker panics at its round …` |
| `cut-silent` | falls silent at that round (it ticks the deadline's injected clock, so no test waits on the wall clock) | `FIREWALL-REFUSED: the device failed: silent past the firewall's deadline` |
| `cut-decode` | returns output that does not decode, from the rep that holds that round | `FIREWALL-REFUSED: the device failed: the device's output does not decode: …` |
| `cut-tree` | samples its first tree at or after that round one leaf larger than the statement's (the CPU device: its trees sample through the salt context) | the frame's stop at the opening round that differs; cut at round 0, the ZK server's `FIREWALL-REFUSED: the proof's exposed value 0 is not the one the firewall passed` may come with it |

`FC_FIREWALL_CUT_AT` counts the table's device rounds over both reps (default 0; a rep's bucket has the count so far as
`firewall_device_rounds`, which leaves out the rounds the host's inner proof sends), and a `cut-` control fires once a table.
The selftest's `zk_worker_cut_MODE` holds each `cut-` control to the rule on the CPU path, beside an honest
session: cut at the table's third, half-way and last device round (`cut-tree` at its first and half-way round), the verifier
gets the honest session's every request up to its `Finish`, in the same order and at the same sizes, with no proof. It
runs through the Lean session loop (`FC_FIREWALL_LEAN`), the verifier logging each request behind it, and skips without
it.

`FC_WIRE_TAMPER` doctors the prover's frames on the wire to the loop instead, and the selftest's `zk_worker_wire` holds
each mode to the loop's own encoding, beside an honest session, the verifier logging a frame that is not its
request's own encoding as `noncanonical`:

| `FC_WIRE_TAMPER` | the prover's frames | outcome |
|---|---|---|
| `round-trailing` | bytes after every `Round`'s count | dropped: accepted, the verifier's every request the honest session's |
| `commit-doctored` | the `Commit` with a `root_F` that is not Σ, a public word and bytes after it | dropped: the same |
| `proof-worker` | each `Proof` one byte of junk | dropped: the same, the proofs the loop's |
| `unknown` | a frame of tag 99 before `Finish` | `PROVER-STOPPED: a request the session loop does not serve`; the verifier gets the honest session's requests up to its proofs, then nothing |

The loop's finish computes each rep's proof file on what its frames hold, not on what the host's request carries. No
request carries the rep's samples (the pads, `τ`'s salt, the pads tree's salts): the finish takes the ones the rep's frame
took at its start (`Finish.drawsOf`). An `open`, `inner` or `rep` request whose passed values, extra claims or coins are
not the rep's frame's (the values it took, `Drive.extras` at the session's link points, the coins the stream answered) is
refused, and the finish ends (`Finish.RepIn.differs`, `Frames.ownOf`), and so are coins that answer an `inner` request's
round but are not the ones the stream answered it (`Finish.coinsWhy`). `final_c`'s slot is the request's, and the passed
values fix it: an `inner` request whose slot is any other fails its `zerocheck c_eval` constraint and is refused before
its inner proof's first round (`Finish.consWhy`), and a `rep` request whose slot is not its `inner` request's is answered
by no proof computed live, so it is refused.
`FC_FINISH_TAMPER` doctors the session's last rep request in time (`salts-early` and `salts-again` ask another finish for
a rep's salts instead), and the selftest's `zk_worker_finish` holds each mode to that, beside an honest session:

| `FC_FINISH_TAMPER` | the last rep request | outcome |
|---|---|---|
| `vals-chosen` | the first passed value plus one | `finish: table T rep R's request: its passed values are not the ones the frame took`; no proof leaves, and the verifier gets the honest session's every request but its proofs, `Finish` included, and does not accept |
| `coins` | the first round's first coin plus one | `… its coins are not the ones the stream answered`; the same |
| `final-c` | `final_c` plus one | `… proof file is only the one its inner request computed live, and none answers this request`; the same |
| `salts-early` | none doctored: at table 0 rep 0's first round a finish of the worker's own asks for the rep's salts for levels ≥ 1 | `… its salts for levels ≥ 1, asked for before its algebraic phase ended`; the session goes on on its own finish, and the verifier gets the honest session's every request and accepts |
| `salts-again` | none doctored: once the session's finish generated table 0 rep 0's salts for levels ≥ 1, a finish of the worker's own asks for them again | `… rep 0's salts for levels ≥ 1 were drawn already, on another finish`: the loop keeps a rep's salts once; the same |

The inner proof's rounds that leave are the frame's own: the loop places each at `Frame.innerSched`, the frame's own round
(`ε`, `β`, its samples and the round's coin count) on the finish's round (`final_c'`, `τ`'s commitment), which the loop's
finish hands the frame before Rust relays it (`Frames.Sessions.innerRound`), and the frame's check holds the relayed round
to it byte for byte; a round the finish has not computed is none the frame takes (`Frames.roundStep`).
`FC_INNER_TAMPER=J` relays table 0 rep 0's inner round `J` with its last value byte flipped, and `FC_INNER_TAMPER=final-c`
sends the session's last `inner` request in time with `final_c` plus one, so the finish computes no inner round of that
rep. The selftest's `zk_worker_inner` holds each mode to that, beside an honest session, which is accepted:

| `FC_INNER_TAMPER` | the relayed round | outcome |
|---|---|---|
| `0` | `final_c'` | the frame's stop at the round, its own round leaving in its place: `PROVER-STOPPED: the Lean frame stopped table 0 rep 0`; no proof leaves, and the verifier gets the honest session's every request but its proofs, `Finish` included, and does not accept |
| `1` | `[ρ, σ]` | the same |
| `2` | `τ`'s commitment | the same |
| `3` | `Y` | the same |
| `final-c` | none: the finish refuses the rep's `inner` request (`its zerocheck c_eval constraint fails`), and the ZK server fails the rep and sends its four inner rounds as no device round | the frame fails each, its own round leaving in its place: `PROVER-STOPPED: the Lean frame stopped table T rep R`; the same |

The selftest's `zk_worker_pads` holds the worker's view to the pads' absence, beside an honest session: under
`FC_FIREWALL_TAMPER=steer` the worker sets each rep's first masked value to a word of its own, in its round and its proof
alike, which a worker holding the pads would send to set the word that leaves. In both sessions no ZK-served rep generated
pads of its own and the word reaches the verifier in no request; the steered session's constraints fail on the pads, so
it stops at its `Finish` with no proof, the verifier getting the honest session's every other request at its sizes.

The outer phase's rounds are the loop's own, so it has no transcript for the contract (`flock-firewall contract`)
to decide after the fact.

The inner phase's controls are V*'s: `rec_inner.py forge --what top` (a clear top that does not open the salted one the challenger's
live-coin part received) and `--what unsalted` (a record of clear tops), refused `FIREWALL-REFUSED: tree top: …` by `rec_vstar`.
`85-rec-reprice.sh` runs both phases (`STEP=icheck`; `STEP=oprove MODES=firewall WARM=0 RUNS=1 CONTROLS=L6`, one
session per staged statement, and `MODES=dev,host` in a job of its own for the ZK server's cost).

### 10.5 What a stop reveals: the whole recursive session

A recursive session is the inner phase's session and the outer phase's sessions, one per V* statement (at K = 4096 eight: the
levels L0–L6 and the algebra). With §10.1 and §10.3, every message of a session that reaches the auditor is fixed, and so is
their order: `Hello`, `Commit`, `Link`, each stream's `Open` and rounds, the proofs, `Finish` (`Ping` and `Register` change
no record: the inner ZK server answers `Register` itself, with the full sample the record already holds). An inner `Open` may come any
time before its stream's first round, but the ZK server passes it to the challenger only at that round (§10.3), so where it
came never reaches the auditor. A `Commit`, `Link` or `Finish` out of its place is refused by the ZK server before any call
on the challenger (§10.3), so it reaches the auditor as a plain stop. An outer session through the Lean session loop
(§10.4) is fixed the same way: `Register`, `Hello`, the first rep stream's `Open`, `Commit`, `Link`, then each rep stream's
rounds and the next one's `Open`, the proofs, `Finish`, each frame that carries no coin once, at its place; anything else
is refused before it reaches the auditor, a `Ping` never leaves, and after a stop or the session's `Finish` nothing
more does. What a worker still chooses is
whether and where a session stops: a refusal it provokes, or a session that ends without `Finish` (a crash, a hang, a cut
connection). The auditor sees every stop:
- inner phase: `rec_live` writes every session it serves, with `end` `finish` or `stop` (a refusal and a session that ends
  without `Finish` alike, never why), and counts it toward `--sessions` (§10.3);
- outer phase: `flock-circuit serve` writes every connection's record, finished or not, and counts it toward `--sessions`.
  Without the ZK server's relay, the outer prover sends `Finish` after a refusal and a crash or a cut does not, so an outer
  record also shows which of the two ended it. Through the relay a stop the ZK server makes reaches serve as a cut, never
  `Finish` (below), so the record shows where the session stopped and not how; a rejecting verdict is serve's own;
- not the ZK server's own record. `record.jsonl` (`verity_flock.firewall`, `firewall/v0`, in the ZK server's state directory)
  stays with the developer, beside the ZK server: it holds more than the auditor's records do (each stop's `why`, every
  `refused` attempt with its part, every `wait` for the slot, each interior registration's own key, §10.2). The auditor
  sees serve's and `rec_live`'s records only.
  `firewall.json`, which `rec_live` writes beside each inner session's record, names the recursive session and nothing
  else.

**A stop ends the recursive session** (`verity_flock.firewall`). `rec_live --firewall DIR --parts L0,…,alg --budget b` opens a
recursive session for each inner session it serves, with its V* statements in their order, public and fixed from K (L0–L6
and the algebra at K = 4096), recorded in the `open` event (`firewall.json`, beside the session's record, names it), and
records its finish or its stop. Every outer session reaches the challenger through the ZK server's relay (`python -m
verity_flock.firewall relay --state DIR --session ID --part NAME --to SERVE`, between `flock-circuit prove --zk` and
`serve`), which serves it only while its recursive session is open, its inner session finished, and nothing of it stopped,
and only for the recursive session's next statement not yet released, once. A request for any other statement (a later
one, or a released one again) is the recursive session's stop at that next statement, never opened: a statement the worker
skips ends the recursive session where it stands, and no statement is served twice. The first stop, inner or outer, is the
recursive session's one failure (`stop`, recorded once, and only of a recursive session opened, so that each stop counts
against its stream). After it, and after the last statement's release, the relay
refuses every V* statement of that recursive session before anything reaches the challenger: the worker's requests are
answered `FIREWALL-REFUSED: recursive session …`, its `Finish` with a verdict that does not accept, and the record shows
`refused`, which is not a failure (the challenger sees nothing of it). So no outer proof of a stopped recursive session is
served or released. An outer session stops when the worker cuts it, when the worker sends `Finish` before all of the
session's proofs (the send check refused: `flock-circuit` then sends `Finish` without its proofs, after every round of the
schedule, §10.1, and the relay forwards `Finish` only after the session's 2 proofs, `--proofs`), when serve's verdict
rejects it, or on any fault of the relay's. In all but the
verdict the relay cuts the challenger's connection without forwarding `Finish`, and the worker gets the relay's own verdict.
A refused or stopped session's worker is answered after the session has left the slot, so a worker that keeps talking
holds up no other session.

**The release side is Lean's** (`core-firewall session`, `Core.Firewall.Loop` in `verity/protocols/recursion/verifier/firewall`).
`flock-circuit prove --zk --firewall-state DIR --session ID --part NAME`, behind its send check (`FC_FIREWALL=1`, §10.4),
starts it as its Lean session loop (§10.4's `flock-firewall session` with a hold on each session, `Session.Hold`;
`FC_CORE_FIREWALL`, else this checkout's build), and the loop keeps `verity_flock.firewall`'s record byte for byte
(`record.jsonl` read whole and appended under `record.lock`, `slot.lock` the outer slot, both `flock(2)` locks):
`live/src/firewall_record_vectors.json`, which `test_the_record_is_the_shared_vectors_byte_for_byte` and
`test_firewall_release_agreement.py` run. Like the relay, the
loop answers a session's `Ping`s itself, and at its first other request takes the slot, then records the slot and the
admission in one read of the record (`Core.Firewall.seat`: the statements' order is enforced there), before anything of
the session reaches the challenger. Given the salt key, it admits the statement's registrations at the challenger's
answer to `Register` (`Core.Firewall.seatReads`, §10.2), before the worker receives the draw. It relays the session's
rounds as §10.4 says, holds its `Finish` and the loop's own
proofs, and at `Finish` decides (`Decision.after`, at the wall clock's time): the loop's own refusal of the session is a
stop, as is a `Finish` with fewer than the session's proofs (two a table, the loop's finish's: where a check refused, the
finish computed none) and a missed release time. A session that goes waits for its release time; its proofs, then `Finish`,
leave, and serve's verdict is its release or its stop. At each decision the loop writes one line to its stderr,
`FIREWALL-TIMES {"session", "part", "ready_ms", "release_ms", "outcome"}`: when it decided, the release time it holds
the proofs to (null on a stop) and `go` or `stop`, by which a run times its windows (pod 85's `RELEASE=1`,
`rec_inner.py release-report`). `verity/protocols/recursion/verifier/firewall/tests` holds these stages to
Python's record and to the cases frozen from the Rust release side the loop replaced (`tests/fixtures/`). Every stop the
loop makes, and a fault of its own, cuts the challenger's connection before `Finish`, and the worker gets the loop's own
verdict; a prover killed while its session holds the slot is that session's stop, which the loop records, or, where the
loop ended with it, the next session to take the slot, if its recursive session was opened. A session cut before its first request other than
`Ping` records nothing, and the statement stays next: no message of it has reached the challenger. `flock-circuit` proves
one session (`--warm 0 --runs 1`: each statement is served once), refuses `FC_VERIFY_AHEAD` (the slot is held until each
session's verdict), and reports the outcome the loop recorded. `verity_flock.firewall` stays as the inner side
(`rec_live`'s opens, finishes and inner stops) and as the tests' relay. These decisions across sessions (whether a
recursive session's next outer session goes on, and when its proofs leave) are outside `FirewallComputes`, in either
language: that assumption covers the values one session's transcript reports the ZK server fixed. The Lean loop
decides them on the record, session by session, before anything leaves; the tests' Python relay decides its own unchecked,
and refuses a schedule.

**Outer proofs as independent sessions.** AGENTS.md's rule: each staged statement
is fixed before any session flips coins and gets exactly one session; each session flips its own live coins in its own
ZK server pass; no statement reads another session's transcript or coins; every session is recorded and every verdict counts.
Sessions may therefore run at once (pod 85's `STEP=oprove`, `OPROVE_JOBS`, which enforces the rule). This relay is stricter:
it has one slot across streams and protocols (`slot.lock` in its state directory, which every relay process of the ZK server
shares). An outer session takes it at its first request other than `Ping` and holds it until the session is released or
fails; a session that asks meanwhile waits (`wait`) and takes it after, so in the record each `slot` is followed by its
session's outcome before the next `slot`. Through it the auditor's outer sessions run in sequence, never interleaved, the
shape soundness composes (`Multi.seqC`);
sessions run at once rest on the provisional batch until a composition for them is proved. The slot orders sessions but
doesn't choose which goes first: across streams the worker picks the interleaving. Rule 3's schedule fixes it: each
scheduled outer session's proofs leave at its own public time, so the interleaving across streams is the schedules', not
the worker's. Through the Python relay, which keeps no schedule, it is the worker's.

**The bound.** A session of `M_p` messages ends finished, never opened, stopped after its `k`-th message (`k < M_p`), or (an
outer session) rejected by serve's verdict. Without the relay, an outer stop also shows whether `Finish` came. Through the
relay it does not, and a session cut before its first request is one never opened (the relay connects to the challenger only
at that request), so every session has at most `M_p + 1` stops. Counted from serve's and the proxy's records of one recursive
session at K = 4096 (2 reps a session): the inner session's 286 messages (139 rounds a rep) and the outer sessions' 268, 266,
266, 242, 242, 236, 232 and 288 (L0–L6, the algebra), 2326 in all.

| a stop | what it reveals, at most | at K = 4096 |
|---|---|---|
| ends the recursive session, its statements and their order fixed before it (only the next unreleased one served, a statement never opened a stop where it stands) | `log₂(1 + Σ_p (M_p + 1))` | 11.19 bits (`log₂ 2336`) |
| ends the recursive session, the statements and their order left to the worker (a ZK server that only ends it at its first stop) | `log₂((M_inner + 1) + 2ⁿ + 2ⁿ⁻¹ Σ_outer M_p)` for n outer statements, plus the order's `log₂ n!` | 18.00 bits (`log₂ 261663`), plus up to 15.30 (`log₂ 8!`) |
| ends its own session only (each V* statement against its own `serve`, no relay) | `Σ_p log₂(s_p + 2)`, `s_p = M_p` stops inner and `2 M_p` outer | 80.1 bits (72.1 = `Σ_p log₂(M_p + 1)` without the outer's `Finish` bit and the never-opened sessions) |

The first row holds wherever the inner phase runs `rec_live --firewall --parts --budget` and the outer phase runs through the
ZK server (`flock-circuit prove --firewall-state`, or the tests' relay; `tests/test_firewall_release.py` and
`verity/protocols/recursion/verifier/firewall/tests`, a rule a test). The rules as `admit` and `Decision.after` run them are proved
properties (`Properties.Firewall`): only the next statement not yet released is admitted, and only before any stop; a
statement asked out of its turn is the stop at the next one; the first stop ends the recursive session; proofs leave
only on time. The first row's count from them is not yet a theorem.
Ending the recursive session at its first stop isn't enough for
it. If the worker chose which V* statements come and in what order, each outer statement would be released or never opened at its choice, with at most one
stopped, and the order in which the challenger's `serve`s are reached would come on top, if the challenger can see it: the
second row. So the statements and their order are fixed when the recursive session opens, and `admit` serves only the
next statement not yet released, once: a statement never opened is a stop where it stands, and nothing after it is
served. `85-rec-reprice.sh` runs neither ZK server: it proves each V* statement against its own `serve`, so its sessions are
the third row's.

**A stream stops at its failure budget.** `rec_live --firewall DIR --budget b` opens each recursive session on its stream
with the failure budget b, a public parameter of the deployment and at least 1 (`firewall.open` and Lean's `openSession`
refuse a lesser one, at which every open would be refused), the same at every open of the stream (`firewall.open` refuses
another). `rec_live` refuses a ZK server without a budget, or with one below 1, before it listens: a stream without a
budget bounds none of its recursive sessions. A failure is a recursive
session's stop (a missed time and a statement never opened included), a refusal of a recursive session that had
neither stopped nor released its last statement, or a registration a statement of the stream passed over (rule 5, §10.2),
on a stream that had not stopped; at the b-th, the ZK server refuses every
outer session of the stream and every open, and `rec_live` closes the next connection before anything is served. A
recursive session is unended while it has neither stopped nor released its last statement, and the ZK server refuses an
open while the stream's unended recursive sessions and its failures together reach b, so at most b of a stream's recursive
sessions ever have a statement not released: each is unended, or stopped and so a failure, since a stop is recorded only
of a recursive session opened (`card_unrel_le_budget`, `Proofs.Flock.Recursive.StopCountBudget`, at every order of the
record's ops `openSession`, `finish`, `stop`, `release`, `slot`, `admit` and `wait`, and `card_unreleased_le_budget` at
every run of `StopCount.run`; rule 5's `register` and `admitReads` are not yet among the ops the proof runs: they open
nothing, their stops and skips are failures like any other, and a `known` event is none). A recursive session the worker
abandons, asking for no statement after some release, ends where it stands for the challenger (its next proof never
comes, `late`); the ZK server records no stop until a request comes, but the session stays unended and holds one of the
b. b is the challenger's stream's `budget` (`Stream(budget=...)`,
`verity/protocols/spot/verifier/service.py`, bound into the stream's digest), and the two must agree in three
places: the same b, the same stream (one ZK server stream a challenger stream), and one recursive session a proof session
of the challenger's: one registration's, or the consecutive registrations one `Stream.session` covers (at most the stream's
`rounds`, at most 4). So each ZK server failure but a skip is a `reject` or `late` there for each round the session
covers, and a skip, which the challenger never sees, only brings the ZK server's end sooner: the challenger's count is
never below the ZK server's stops and refusals, and it holds at most `budget` minus its failures registrations open, so
neither ends past b. An honest stream, which passes over no registration, is never refused an open: each of its unended
recursive sessions covers a registration still open, since the relay releases a session's last statement before the ZK
server delivers the session and the challenger closes its rounds, so at an open fewer than min(`rounds`, b) are unended
(`test_an_honest_stream_is_never_refused_an_open`, in Lean and Python, at `rounds` 1 to 4 and b 1 to 4). b is how many
failures stop the stream and, at min(`rounds`, b), how many of an honest stream's recursive sessions are unended at once.
A stop shows the challenger a cut or a missing
proof, which is `late` or `reject`; a session the ZK server refuses after its stop sends nothing, and is no further failure
on either side.

**Proofs leave at public times.** `rec_live --parts L0,...,alg --release-windows-ms w0,...,w7` gives each recursive
session a release schedule, recorded at its open: `start_ms`, the time its connection was accepted, and one public window
a statement, in the order `--parts` fixes. Statement j's proofs and `Finish` leave at `start_ms + w0 + ... + wj`, never
earlier (a session ready early waits) and never late (a session not ready by then is a stop, a missed time). The schedule is fixed
before the session and depends on nothing secret: not on how long the inner session or any outer one took. Only the Lean
release side keeps the times; the Python relay refuses a scheduled session rather than release it early. The release of part
j follows part j − 1's verdict by its session's end to end, so a window is sized from honest timed sessions as
`w_j = 1.5 × (max e2e_j + max gap_{j−1})` rounded up to a second, `gap` being a slot's hold beyond the session
(`tp_s − e2e_s`, the verdict).

Two limits. L0's window counts from the connection's accept, so on top of L0's session it must cover the inner session and
the V* stage (`InnerFold`). And the slot is shared: two recursive sessions whose outer phases overlap in time take turns at
it, so the later one misses its times. That is a failure, never an early release, but an honest operator must space
recursive sessions so their outer phases don't overlap, and nothing enforces that yet. A ZK server on cores of its own would
take the recheck and serve's verify off the prover's cores, so co-tenant load could not turn an honest session into a
missed time and spend the budget.

**The bound over a stream.** Over a stream of n recursive sessions with budget b whose statements read r registrations,
rules 1–5 leave the challenger: which recursive sessions failed and which registrations were passed over (at most b in
all, by rule 4), for each failure j, the form it took (one, at one position of the order fixed at open, by rule 1), and
for each registration i admitted, which of its epoch's m_i sessions it is (rule 5, §10.2; m_i = 1 on the service of
record, so none). Every proof that leaves,
leaves at its public time (rule 3), one at a time (rule 2), so the interleaving across streams is fixed too. That is at
most

`log₂ Σ_{i≤b} C(n + r, i) + Σ_j log₂ q_j + Σ_i log₂ m_i` bits,

with q_j the number of forms failure j can take, one for a registration passed over, and the last sum 0 on the service of
record. The registrations' part is
`Verity.Guarantees.RegisteredStreamLeakage`'s; the rest is counted here. The figures below are a stream with no
registrations (r = 0). A refusal of the outer send check is one symbol: it stops its session at
the session's `Finish`, after every round of the schedule (§10.1), so a stream whose failures are all the ZK server's reveals
`log₂ Σ_{i≤b} C(n, i)` bits (at n = 1000: 9.97 at b = 1, 35.3 at b = 4), and each session one bit. Any other failure
is not one symbol today: the challenger sees where its session was cut, inner or outer (a cut, crash or hang of the
worker's, the inner phase's refusals, a missed time), so `q_j ≤ 1 + Σ_p (M_p + 1) = 2336` at K = 4096, the first row's
11.19 bits, and the bound that holds for those is `log₂ Σ_{i≤b} C(n, i) + 11.19 b` bits (for n = 1000: 9.97 + 11.19 =
21.2 bits at b = 1, 35.3 + 44.8 = 80.0 at b = 4). An outer session is interactive: the challenger answers each message
with its coins, so every message before a cut has reached it, and so has where the cut fell; the ZK server takes the cut's place out only for the stops it makes itself, by releasing
the schedule in their place. Not counted in the bound: timing
inside a session. The release time fixes when each outer session's proofs and `Finish` leave, not the pace of its rounds
before them, which reach `serve` as the prover makes them, nor when a stop's cut comes (a missed time's comes when the late
session is ready). Holding every cut until its statement's release time is the ZK server's to do and would take the cut's
time out; it is not done yet.

**The record fails closed.** Each call reads `record.jsonl` under `record.lock`. The Lean release side reads it whole
every time (`Core.Firewall.Record.read`). The Python relay reads only the lines appended since what the rules read of it
was last brought up to date, which it keeps beside the record in `index-py/`, since each of its processes serves a
session or a few. It reads the record again from its start when it was replaced, cut shorter or no longer holds the last
bytes read, when the machine has booted since its index was written, when a writer of the index ended mid-write, or when
a file of it won't read; it reads the record whole when its last line has no newline yet, a line is one the ZK server
doesn't write or a name is over 100 bytes. An edit further back that leaves the record no shorter isn't seen by it, nor
is a file of the index edited by hand that still reads. A
torn last line (a writer killed inside its append) fails every call of either ZK server, so nothing is served or released
until the developer repairs the record; neither skips an event it can't read, since a skipped `stop` would serve a
stopped recursive session. The Python relay reports such a fault as its connection's outcome `error`.

**Retries.** Every stop is recorded and counted, in both phases, so the auditor knows how many times a recursive session was
run, and each run it allows reveals its own ending: `r` stopped runs before an accepted one reveal at most `r + 1` times the
row's bound. The stream's budget bounds them: a stream has at most b failed recursive sessions before it stops, whoever
asks for the next run, so the bound over the stream counts every retry, and a stream whose ZK server stops b times runs
nothing more. b is a public parameter of the deployment, so the bound is quoted at the deployed b.
