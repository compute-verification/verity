# Ligero-style direct-constraint argument for the transition unit (lane b-ligero, track B)

Status: NON_ZK_PROOF_DIAGNOSTIC without `--zk`; with `--zk`: COMPLETE_ZK_BACKEND in `--mode interactive` (sections 8b + 8c as repaired by lane b-zk-fix after the red-team-zk findings F1-F3: `note:r20-proof/red-team-zk/20260922T1102Z-finding-redteam-zk-urgent`, `note:r20-proof/b-zk-fix/20260922T1142Z-report-b-zk-fix`), COMPLETE_HVZK_BACKEND (NIZK in the programmable ROM) in `--mode fiat-shamir`. In every ZK row: the current statement publishes `a, b, y16`, so privacy is vacuous for it; the claim is about the construction. Implemented in this directory (Python/torch); measured on `vy-g3` (NVIDIA L40S).
The relation is the frozen WP2 checker (`verity_numerical.checker`): every predicate of `check_step_expr` at
`params.REAL` is constrained; nothing is dropped (section 5 lists the two places where a predicate is *restated*
and argues equivalence).

## 1. Statement and shape

A proof establishes, for `B = l` transition units at once,

    y[i] = tc_dot(c[i], a[i, 0..16), b[i, 0..16))        i < l

with `c[i]` an FP32 word, `a, b` BF16 words, `y[i]` the outgoing FP32 word (an infinity when a group saturates).
All of `(c, a, b, y)` are public; the prover's witness is the checker's hints (`gen_step`) plus the derived rows.
`l` is a power of two (4096 in the measurements; a batch is padded with the trivial unit `c = a = b = 0, y = 0`).

**Witness matrix W (m x l)**: one *column per unit*, one *row per witness variable* of the unit. Rows are what the
encoder encodes (each row is a message of `l` symbols), so the unit-uniform structure makes every constraint a
constraint *on rows with scalar coefficients*, which is exactly what Ligero's row tests need (section 4). At
`REAL`, `compile_unit` gives

    m = 3372 rows:  99 pin | 92 hint | 170 product | 2401 bit | 610 selector
    338 linear constraints (102 with a public right-hand side), 3246 quadratic constraints

The census's "358 mul / 137 lin / 306 lookups" becomes, with lookups and range checks as explicit columns:
every bit row carries one quadratic constraint (`b^2 = b`), every selector row one, every product row one, and
each range check / one-hot one linear constraint. So 3246 quadratic constraints are mostly booleanity.

### Row classes (`compile.py`)

* `pin` -- a decode of a public word: for `c`: `(s, e_hat, M)` of `Fp32Decode`; for each `a[i]`, `b[i]`:
  `(s, m, e)` of `Bf16Decode`. The verifier computes these itself from the public words (O(1) per unit, i.e. reading
  the input) with the gadgets' own arithmetic (`unit.public_vectors`, cross-checked against
  `unit.reference_decode`), and the constraint `W_row = pub` (a linear constraint with a public right-hand side)
  binds the committed row to it. Committing the pins (rather than treating them as per-column constants) keeps
  every constraint scalar-coefficient: a hint multiplied by a public per-column value is a product of two rows.
* `hint` -- a value of `gen_step`: `prod[i].z`, `g.max.E`, `g.max.sel[j]`, `g.acc_align.{q,r}`,
  `g.prod_align[i].{q,r}`, `g.sum.{sgn,z}`, `g.lead.l`, `g.norm.{clamp,M,r}`, `g.out.{z,ovf}`, `pack.{hidden,low}`.
* `prod` -- `z = x * y` for two non-constant affine expressions the gadget code multiplies (memoised).
* `bit` -- the bits of a range-checked expression (`rng`), `sel` -- a one-hot selector of a lookup (below).

## 2. Field: BabyBear, and the integer audit

`p = 15 * 2^27 + 1` (31 bits, 2-adicity 27). Reasons: note:r20-proof/tensor-cost/20260922T0450Z-report-tensor's 1.6x for B from 4-limb base products, and the
RS encoder needs a 2^14 (here) to 2^17 root of unity, which BabyBear has and Mersenne-31 does not without a circle
FFT. Challenges: `D` *independent* BabyBear combinations (`D = 6` at 2^-128, `D = 4` at 2^-80, the degrees of
`accounting.first_extension`); each of Ligero's tests is run `D` times with independent base-field coefficients,
which gives the field term `1/p^D` exactly as an extension of degree `D` would, and needs no extension
multiplication anywhere (every test is linear in the challenge).

The price of a 31-bit field is that the checker's integer identities do not automatically hold in the field:
`2^32 > p`, `x * ipow` with `x, ipow < 2^25` wraps, and even `|S| < 2^30` sits 0.9 bits below `p`. The compiler
therefore *audits every linear constraint over the integers*: with every row's integer interval established by the
range constraints themselves (bits are `[0, 2)`, a hint's first range check *defines* its integer, a product row's
interval is the product of its operands' intervals), the difference `LHS - RHS` of every linear constraint must
lie in `(-p, p)`, so that the field identity implies the integer identity. `CompileError` otherwise. Interval
arithmetic alone is too loose for the masked forms `k X + c (bit X)` (SignedSum's `(1 - 2 s) q` and `v = S - 2 sgn S`,
StateOut's `e_hat`), so the audit evaluates those at `bit = 0` and `bit = 1` (`_interval_peel`). Four
products cannot be bounded by their operands' ranges and are accepted from a stated theorem (`audit_log`, 22 lines):

| product | naive bound | accepted | why |
|---|---|---|---|
| `AlignState q * pow` | 2^50 | 2^25 | `lt_pow(q, pow, W)` states `q < 2^(W - min(d, W))` |
| `AlignProduct q * sr` | 2^32 | 2^16 | `lt_pow(q, sr, 2 sig)` states `q < 2^(2 sig) / sr` |
| `Normalise div * M` | 2^31 | 2^30 | driver check `M < 2^min(acc_sig, s_bits + k)` (theorem: `M = floor(v 2^k)`, `v < 2^s_bits`) |
| `Normalise v * mul` | 2^54 | 2^30 | driver check `v < 2^min(s_bits, acc_sig - k)` (theorem: `M < 2^acc_sig`) |

The two driver checks are consequences of the relation (an honest witness always satisfies them), so the accepted
set is unchanged; they cost two `lt_table` checks per group.

## 3. Reed-Solomon code and commitment

Row `W_j` (l symbols) is read as the evaluations of a polynomial of degree `< l` on `H = <w_l>`; its codeword is its
evaluations on the coset `g <w_n>` with `g = 31` (`n = 4 l`, rate 1/4 as in note:r20-proof/tensor-cost/20260922T0450Z-report-tensor's B row). `H` and the coset are
disjoint, so `Z_H(x) = x^l - 1` is invertible on every codeword position (used by the quadratic test) and
`Enc(constant) = constant`. `encode` = INTT_l, zero-pad, scale by `g^k`, NTT_n -- a batched radix-2 int64 NTT in torch
(`field.py`); the INT8 matmul-DFT variant was not measured (section 8).

Commitment: Merkle tree over the `n` columns of `U = Enc(W)` (m x n), leaf = H(column as m uint32), Blake3 on the host
(PLACEHOLDER; SHA-256 fallback; the hash is named in the proof). The column transfer (`U^T` to host) is timed
separately (`split.transfer_seconds`).

## 4. The tests (`protocol.py`)

Transcript (Fiat-Shamir, SHAKE-256 as the challenge expander -- placeholder):

    P -> V   root
    V -> P   r (D x m), rho_lin (D x L), rho_quad (D x Q)      from H(root)
    P -> V   w = r W (D x l);  h (D x (l - 1))
    V -> P   cols: t distinct column indices                   from H(root, w, h)
    P -> V   U[:, cols] (m x t), Merkle paths

Two verifier coins -> `rounds.sequential_depth = 2` (the calculator's B row has 3: one more for the lookup helper
message `z`, which a direct-constraint B does not have).

1. **Interleaved proximity**: `Enc(w)[cols] == r U[cols]` for each of the `D` combinations.
2. **Linear constraints** `sum_j a_qj W_j + c_q = pub_q` (pub_q a public vector or 0): the verifier forms
   `alpha = sum_q rho_q a_q` (D x m) and `beta = sum_q rho_q (pub_q - c_q)` (D x l), and checks
   `Enc(beta)[cols] == alpha U[cols]`. *Nothing is sent*: with the right-hand sides public, the message-domain
   value of the combination is fixed, and the verifier encodes it itself (one INTT_l + NTT_n of `D` rows).
3. **Quadratic constraints** `A_q(W) o B_q(W) = C_q(W)` (affine `A, B, C`): the prover forms
   `P0 = sum_q rho_q (A_q(U) o B_q(U) - C_q(U))` (D x n) -- degree `< 2l - 1`, vanishing on `H` when the constraints
   hold -- and sends the quotient `h = P0 / Z_H` (degree `< l - 1`, as coefficients). The verifier checks
   `sum_q rho_q (A_q(U[cols]) o B_q(U[cols]) - C_q(U[cols])) == h(eta_c) Z_H(eta_c)`. Sending the quotient rather than
   `P0` folds "P0 vanishes on H" into the degree bound.
4. Merkle paths of the opened columns.

The prover's cost of 2 and 3 is one pass over the sparse constraint tables applied to `U` (index-gather, ~6 000
terms), one Hadamard on `Q x n`, `D` reductions `Q x n -> n`, one INTT_n. No extension arithmetic.
*Split form (lane b-ligero2, `witness.quad_p0`)*: the same `P0`, computed as the booleanity part
`sum_{j bit/sel} rho_j (U_j^2 - U_j)` in one fused pass over those rows (3196 of the 3387 constraints; no `Q x n`
intermediate) plus the 191 general constraints through the sparse tables of their subset. Bit-equal to the form
above (it is the same polynomial); 122 -> 29 ms per proof at `n = 65536`, peak memory 13.7 -> 8.5 GB at 170 VUs.
Under ZK (section 8b) both sides add the mask contributions to `P0` before / after the division by `Z_H`.

Every challenge is `H(statement, transcript prefix)` with the statement digest computed by the verifier from its own
configuration (lane b-ligero2, `statement_digest`; the pre-fix protocol derived the challenges from the transcript
alone and was forgeable by choosing the claimed words after the challenges -- README.md "Second pass").

## 5. Where the checker is restated (relation unchanged)

1. **Decodes of public words** (`Fp32Decode`, `Bf16Decode`): the word identities `s 2^31 + t 2^23 + f = c` cannot be
   stated in BabyBear (`2^32 > p`). Instead the decoded components are pinned rows the verifier recomputes from the
   word with the gadget's own `gen` + `check` (integers), including the domain checks `t != all-ones`
   (`x.* range`, `acc.t`, `a.t`, `b.t` -> the verifier rejects). Accepted set: identical (the decode is a function).
2. **The output word**: `Pack`'s `y` is compared component-wise (`y.s`, `y.t`, `y.f`), never as the 32-bit word.
   Equivalent because whenever the finite branch is live (`any_ovf = 0`, `z = 0`), StateOut's `ovf_proof` gives
   `e1 <= e_max`, so `fld <= 254` and every component is in its field's range, and the infinity branch sets the
   components directly; then word equality <=> component equality.
3. `lt_pow(x, ipow, bits)` (`x < 2^bits / ipow`) is realised as `pw - 1 - x in [0, 2^b)` with `pw = 2^bits / ipow`
   (an affine function of the same selectors) given `x in [0, 2^bits)` -- no product, so no wrap. Equivalent over the
   integers. When `2^bits + 2^b` would reach `p/2` (the 30-bit `v`), the check is split over the two halves of `x`'s
   bit decomposition (`lt_table`).
4. **Lookups** (`POW`, `ALIGN`, `LEAD`, `NORM`): a one-hot selector over the *distinct rows* of the table (26, 27, 31,
   32 selectors), `sum sel = 1`, `key = sum sel_g k_g` when every group is a single key; for `POW`/`ALIGN`, whose top
   rows repeat (`d >= W`, `d >= prod_bits`), `key - sum sel_g k0_g >= 0` (bits) and `= 0` unless the top group is
   selected, plus the table-domain range check on the key. Outputs are affine in the selectors. This is the lookup
   argument "you also implement": it proves key membership in the table domain and the row's correctness.
5. **Range checks**: bits of `x - lo`; for a width that is not a power of two (`t_finite`, the `E` range), also the
   bits of `hi - 1 - x`. Memoised per expression.

## 6. Soundness accounting

`security.accounting.ligero_error` (Appendix C, `e = (n - k)/2`, uniform combinations), with `k = l` (no ZK padding;
`validate()` needs `l < k`, so `k = l + 1` stands in, conservative; `config_for` sizes `t` and `D` at the stand-in too,
so the reported bound meets the target, `archive/ligero-verify/discrepancies/d1.md`, BV-D1) and `F_challenge = BabyBear^D`:

    l = 4096, n = 16384, t = 189, D = 6:
      irs_query (1 - e/n)^t = (5/8)^189 = 2^-128.1    linear_query ((k+l)/n)^t = 2^-189    quadratic_query (2k/n)^t = 2^-189
      irs_field (n+1)/p^6 = 2^-172      linear_field, quadratic_field 1/p^6 = 2^-186
      total 2^-128.1
    2^-80 sensitivity point: t = 118, D = 4 (fields 2^-110..-124).

`t` is `ligero_queries_for(target, n, l, l)`. The linear test sends nothing, so its query term is at most the Appendix
C one used.  A row-sharing GEMM side (8e) adds the statistical term `fingerprint_collision` (2^-157 / 2^-139 per proof).
A chain of more than three linked rows (the hashed relations: the leaf's carried state) adds `chain_field`: its `E` extra
constraints take monomial coefficients of degree up to `1 + ceil(E / 6)` (`chain.py`), so a violation survives a
coordinate with probability up to that degree x 3 / 2^32 (the coins are uint32 mod p), i.e. `(3 deg / 2^32)^D` per proof
(`chain_term.py`): 2^-165.6 for Poseidon2 (19 rows), 2^-155.4 for SHA-256 (69), 2^-149.5 for the 3-slot BLAKE3 frame (133),
2^-137.4 for BF16 K = 8192 BLAKE3 (549).  `config_for` sizes `t` with it and ligero-verify books it too
(red-team-bligero-real-k A1).  The proximity-gap statement for base-field rows combined by base-field coefficients is Ligero's own
setting (no subfield/extension assumption is needed here, since the combinations are base-field).

## 7. Communication (l = 4096, m = 3372, t = 189, D = 6)

    root 32 B; w: D l * 4 = 98 KB; h: D (l - 1) * 4 = 98 KB; cols 756 B;
    opened columns: m t * 4 = 2.55 MB; Merkle paths: t * 14 * 32 = 85 KB      total ~2.8 MB per 4096 units (~700 B/unit)

The opened columns dominate (Ligero's `t * m`); `m` is what a cheaper range-check realisation would shrink.
Chain (m = 3516, + `q`: D (2l - 1) * 4 = 197 KB): 3.14 MB per 42 VUs (75 KB/VU).

## 7b. The chain: the proof unit (`chain.py`, `vu.py`; `compile_unit(chain=True)`)

A VU is 96 units from `c_0 = +0` plus the `Epilogue` (RN-even cast to BF16). Column `j = 96 v + i` holds unit `i` of
VU `v`; a proof of `l` columns holds `floor(l / 96)` VUs (`l = 4096`: 42 VUs, 64 pad columns = trivial units).
What changes against the unit system:

* The accumulator is no longer public. Its word components `c.s, c.t, c.f` (plus `nz, z`) are **hint rows**, and
  `Fp32Decode.check` runs in-field *without* `acc.word` -- the components are the word (the ranges `s in {0,1}`,
  `t < 255`, `f < 2^23` make it unique; `2^32 > p` is why the word itself is never formed). Same predicates otherwise.
* The output components `y.s, y.t, y.f` stay expressions (no `y.*` pins). The **epilogue** is constrained in every
  column on the unit's own output: `Epilogue.check` with `y32 = hi 2^16 + lo` restated as `f = fhi 2^16 + lo`
  (`fhi < 2^7`), `hi = s 2^15 + t 2^7 + fhi` -- the same integers; `hi2, lsb, carry, rem` as in the gadget;
  `y16 = hi + carry`. Per column this is unconditional (a selector-gated version would raise the quotient's degree
  to `3l`, which the rate-1/4 code does not pay for); it costs 6 hint + ~55 bit rows (3372 -> 3516 rows).
* **Cross-column constraints** (`c_in[j+1] = c_out[j]` on the components, `c_in = 0` at every chain start and pad,
  `y16 = y16_pub` at every chain end) are one linear system `A w = b` over the *flattened* witness and are checked by
  Ligero's **general linear test**: the verifier draws one uniform coefficient per constraint (`7 l` per coordinate,
  `D` coordinates); `r^T A` is a coefficient per `(row, column)`, interpolated per involved row to `R_i(X)` (degree
  `< l`, ~20 rows: the three `c.*` rows and the rows in the `y.*` / `y16` expressions); the prover sends
  `q(X) = sum_i R_i(X) p_i(X)` (degree `< 2l - 1`, `D (2l - 1)` coefficients = 196 KB); the verifier checks
  `sum_{H} q = l (q_0 + q_l) = r^T b` and `q(eta_c) = sum_i R_i(eta_c) U_i[c]` at the opened columns. No rotation, no
  extra openings, no second commitment: the shift `j -> j+1` lives in the coefficients, not in the code. Soundness
  term: the linear test's `((k + l)/n)^t = 2^-189` (Appendix C, `k = l`) plus `1/p^D` for the combination.
* Transcript: `q` is absorbed before the column challenge (`cols = H(root, w, h, q)`). Depth stays 2.
* Verifier: `public_vectors` on the operands only (`a, b` pins; the domain checks), `y16_pub` per VU, and the layout.

Gate (vy-g3, `notes-asset:campaigns/r20-proof/assets/b-ligero/results/gate_vu.json`): 64 `vu-k1536` VUs with the recorded accumulators (two proofs, 42 + 22)
accepted; 52/52 `vu-k1536-neg` rejected (`wrong` at the chain test's sum check, `reject` at the operand domain check
or as a foreign proof); 249 mutated rows + 3 chain-specific mutations (a broken link `c.f[40] != y.f[39]`, `c_0 != 0`
at a chain start, a flipped claimed `y16`) rejected.

## 8. Encoder: int64 NTT vs INT8 matmul DFT (`encoder_bench.py`, vy-g3, m = 3372, l = 4096 -> n = 16384)

    int64 radix-2 NTT encode (used; eager, 7 kernels per stage):     116 ms   (1.9 GB/s of codeword)
    the same NTT with a torch.compile'd butterfly (one kernel/stage):  31 ms   (7.1 GB/s; the prover's encoder now)
    one-stage dense DFT, 16 INT8 limb matmuls (torch._int_mm) only:   52 ms   (139 TOPS; exact = NTT on a slice)
    one-stage + limb recombination mod p in torch int64:             272 ms   (the recombination, not the matmul)
    two-stage 64 x 64 INT8 matmuls only (x4 shifts for n = 4l):        8 ms   (no twiddles / recombination)

The tensor cores win the arithmetic by 2x (one-stage) to 14x (two-stage), but in torch the limb recombination and
twiddle passes are int64 elementwise kernels that cost more than the NTT itself; the matmul DFT pays only with a fused
kernel (recombination in registers). With the compiled butterfly the NTT is 31 ms = 31% of the 0.10 s prover; the
two-stage INT8 DFT (8 ms of matmul) would pay only once its recombination is fused into the same kernel.

## 8b. ZK masking -- implemented (lane b-ligero2; `--zk`; proof class COMPLETE_HVZK_BACKEND)

**As implemented** (`protocol.py`, "ZK masking"): honest-verifier zero knowledge by Ligero's blinding, in the
*coefficient* form, which keeps `n` where it is.

* Row `j` of the witness is committed as `p_j(x) = interp_H(W[j])(x) + Z_H(x) s_j(x)` with `s_j` uniform of degree
  `< t_pad = 256`: `k = l + 256` (not a power of two -- the encoder is INTT_l + coefficient padding + NTT_n, so the
  message length is free), `n = 4 l` unchanged. `p_j` agrees with the payload on `H`; at any `t <= t_pad` points off
  `H` its values are uniform (the map `s -> (Z_H(eta_c) s(eta_c))_c` is onto, `Z_H != 0` on the coset). The
  evaluation domain was already the coset `g <w_n>`, disjoint from `H` (red-team F4: no opened column is a payload
  position; what the unmasked prover leaked was the *interpolant* -- constant rows in the clear at every column, 35%
  of all opened values were `0`, see `redteam_zk.py`).
* Per challenge coordinate `d`, five committed mask rows, all of degree `< k` (so the proximity test covers them like
  any other row; the verifier reconstructs each mask's contribution at the opened columns from the opened mask rows
  and the public `Z_H(eta_c)`, `eta_c^k` -- a dishonest mask is just another committed row, no unchecked freedom):
  `s_prox` (degree `< k`), `s_lin` (`< t_pad`), `s_qa` (`< k`), `s_qb` (`< t_pad - 1`), and -- since b-zk-fix (red
  team F1) -- the two chain mask rows `u_ca` (`< k`) and `u_cb` (`< l - 1`), six kinds (`M = m + 6D` in chain mode).
* Messages: proximity `w_d = r_d . p + s_prox_d` (`k` coefficients, uniform; `r_d` is zero on the prox rows, which
  enter with coefficient 1); linear `v_d = alpha_d . p + Z_H s_lin_d` (`k` coefficients; the verifier checks
  `v_d|_H = beta_d` -- the public right-hand side, folded mod `x^l - 1` and NTT'd -- and `v_d(eta_c)` against the
  opened rows; uniform given `beta`); quadratic `h_d = P_d / Z_H + s_qa_d + x^k s_qb_d` (`2k - 1 - l` coefficients,
  every one masked); chain `q_d = Q_d + u_d` with `u_d = u_ca + x^k u_cb` uniform of degree `< l + k - 1` subject to
  `sum_H u = 0` (the constant coefficient is adjusted; AHIV22 `u_add`): `q` is uniform on the affine hyperplane
  `sum_H q = rhs`, the verifier's check `q(eta_c) = sum_i R_i(eta_c) U_i[c] + U_ca[c] + eta_c^k U_cb[c]` reconstructs the
  mask from the opened rows.  **Pre-fix (red team F1, ACCEPTED-LEAK): `q_d = Q_d + Z_H s_chain_d` -- the mask vanished
  on every point of `H`, so `q|_H = Q|_H` and every intermediate accumulator was recoverable from the transcript
  (16150/16150 at the measured layout).**  Regression: `redteam/run.py` (`q_leak` section) must recover nothing.
* Statement digest absorbs `zk`, `k` and the `mode`; the verifier's shape checks use `k`, `2k - 1 - l`, `l + k - 1`, `M = m + 6D`.
* Randomness: SHAKE-256 of a fresh `os.urandom(32)` seed, reduced mod `p` (2^-31 statistical bias per symbol; the
  simulator's distance). Cost: `m t_pad + D (3k + 2 t_pad + ...)` symbols, ~1.5 M per proof at `l = 8192`.
* Soundness: Appendix C with `k = l + 256`: `t = 199` at `l = 8192` (was 198 for the 49-proof union), `197` at
  `l = 16384` (was 196); `LigeroParams(zk=True)` validates `k > l + t` (Lemma 4.15).
* Test: `redteam_zk.py` -- a proof of real VUs against a proof of the trivial VU at the same layout (what a
  simulator without the witness produces): two-sample chi-square on the opened columns and on `w, v, h, q`
  indistinguishable (`p > 0.3` typical), opened columns uniform, zero opened values equal to a payload value of
  their row, zero constant rows in the clear (unmasked control: 1236 of 1236 constant rows in the clear, 233 k of
  665 k opened values equal to their row's payload).
* What separated this from `COMPLETE_ZK_BACKEND`: the verifier's coins were Fiat-Shamir of the transcript; against a
  malicious verifier (interactive setting) the HM96 commitment to the coins before `root` is needed (one more round,
  depth 3, no arithmetic cost) -- implemented by lane b-zk, section 8c.

The original cost estimate follows (it assumed `k` rounded to a power of two, hence `n = 32768`; the coefficient
form above avoids it -- the measured cost is in README.md).

Ligero's honest-verifier ZK: every row is padded with `t` uniformly random message symbols (`k = l + t`, so an opening
of `t` columns reveals nothing of the row; Lemma 4.15 needs `k > l + t`), and the linear / quadratic / chain test
messages are blinded by random codeword rows (`w`, `h`, `q` each get one masking row per coordinate: `D (m ... )`
extra committed rows, negligible). With `k = l + t = 4285 -> 8192` (the next power of two for the NTT) at `n = 16384`
the rate halves to 1/2: Appendix C then needs `t` such that `(1 - e/n)^t + ((k + l)/n)^t + (2k/n)^t <= 2^-128` with
`e = (n - k)/2 = 4096`: `(3/4)^t` -> `t = 309`, `(2k/n)^t = 1^t` -- **the quadratic test fails at rate 1/2**
(`n > 2k` is required). So ZK at `2^-128` needs `n = 32768` (rate 1/4 again with `k = 8192`): encoding, transfer and
hashing double (`+0.03 s` on the 0.095 s prover at B = 4096: encode 0.031 -> 0.062, hashing 1 -> 2 ms, ~+30%), the opened columns stay `t = 189` (same `(5/8)^189`
with `e = (n - k)/2 = 12288`) but `m t` bytes are unchanged, `w, h, q` double in length (`+0.4 MB`), peak memory
doubles (`6.3 GB`; the 85-VU `l = 8192, n = 32768` run measured 6.6 GB). Alternatively keep `n = 16384` and halve `l` to 2048 units per proof (42 -> 21 VUs): same
per-unit encoding cost, twice the fixed per-proof overhead. Malicious-verifier ZK (the AGENTS.md requirement) adds
a commitment to the verifier's coins before `root` (one more round; depth 3) -- no arithmetic cost.

## 8c. Malicious-verifier ZK: the step-0 coin commitment and the two modes (lane b-zk; repaired by lane b-zk-fix)

Two protocols share the arithmetic; they are **mutually exclusive** (`Config.mode`, bound into the statement digest;
`prove` / `verify` reject the other mode's coin usage; `run.py --mode interactive | fiat-shamir`).

### Interactive mode (`--mode interactive`, the default) -- proof class COMPLETE_ZK_BACKEND with `--zk`

~~~text
 0   V -> P   c1 = H(tag || r1 || s1),  c2 = H(tag || r2 || s2)     r_i, s_i 32 uniform bytes each; H = Blake2b-256
 1   P -> V   root
 2   V -> P   (r1, s1)                     P checks c1; abort (CoinOpeningError, public) on mismatch
              r, rho_lin, rho_quad [, rc]  = SHAKE-256( H("ligero-b/c1|v2" || stmt || c1 c2 || r1) )     -- NO root
 3   P -> V   w, h [, q] [, v]
 4   V -> P   (r2, s2)                     P checks c2; abort on mismatch
              cols                         = H("ligero-b/c2|v2" || stmt || c1 c2 || r2)  (t distinct)        -- NO w h q v
 5   P -> V   U[:, cols], Merkle paths
~~~

Three verifier messages on the critical path: **`rounds.sequential_depth = 3`**.  `verify(coins=own)` is the only
verifier: it requires the transcript's openings to be its own step-0 randomness (a proof from another session, or
one whose coins the prover chose, is rejected: "coins are not this verifier's step-0 coins").  There is no
non-interactive reading of this mode.

**The challenges are functions of the verifier's coins alone** (b-zk-fix, red team F2).  Before the fix the seeds
were `H(stmt, c, root, r1)` and `H(stmt, c, root, w, h, q, v, r2)`: after the rewind the simulator knew `r1` but not
`r` (which depended on the `root` it had yet to build), so the simulator existed only by programming the hash at
`(root, r1)` and `(root, w, h, q, v, r2)` -- the programmable random-oracle model, exactly what step 0 was meant to
avoid.  Now `(stmt, c1, c2, r1)` and `(stmt, c1, c2, r2)` are all fixed before the prover's first message, so
knowing `(r1, r2)` IS knowing `r, rho, rc, cols`.

*Soundness of the coin-only derivation (interactive model).*  The public-coin Ligero analysis (Appendix C, the terms
in `soundness()`) needs the combination challenge `r, rho` to be uniform and independent of `root`, and the column
set `cols` uniform and independent of `(root, w, h, q, v)`.  Both hold: `r1, r2` are the verifier's, sampled before
`root`, hidden from the prover by `c_i` until the prover has committed the message the coin must follow (the
`commitment_hash` hiding term, `2 x q 2^-256`, is the only new soundness term), and the prover cannot influence
them.  `stmt` and `c1 c2` inside the seed are domain separation (the same coins in another session or another
statement give other challenges), not entropy.  Including `root` in the seed bought nothing in this model: `root`
is the prover's message, and a seed that depends on it is not more random for the prover.  The bound is the
statistical interactive bound: **t = 197, D = 6, union 2^-128.05 over 25 x 170** (unchanged from b-zk; the coin
hiding term is in the union).

*Hash assumptions of this mode.*  Collision resistance of Blake2b (coin commitment binding: a second opening is a
collision, `2^-128` at `2^64` queries) and of Blake3 (Merkle); SHAKE-256 as a PRG of the verifier's coin (any
extendable PRF/XOF would do -- the prover never queries it on inputs of its choosing).  Hiding of the coins from
the prover (`c_i = H(r_i || s_i)`, 512 random bits) is booked as a random-oracle term but only needs a hiding
commitment.  **No hash programming; the challenge hash needs no random oracle.**  Exactly one item of the
zero-knowledge argument still needs the random-oracle model (red-team-zk-2 G3): the hiding of the **unopened
Merkle leaves** -- the leaves are unsalted `Blake3(column)`, so the siblings in the opened paths hide their
(masked, high-entropy) columns only as a random-oracle-flavoured property of Blake3.  The interactive ledger row
therefore lists `assumptions = ["hash", "random-oracle (Merkle hiding)"]`.  Salting the leaves (`Blake3(column ||
salt_c)`, `salt_c` uniform 32 bytes, opened with the column) would make the commitment statistically hiding and
remove that item; cost: `+32` bytes hashed per leaf at commit (one extra Blake3 block per column, against `4 M`
bytes of column already hashed) and `+32 t` bytes per proof in the openings (`+6.3 KB` at `t = 197`, ~0.1 % of the
transcript) -- not implemented.

*Simulator (demonstrated: `redteam/simulator.py`, run against the real `verify()` under the real hash).*  `S`
receives `c1, c2`, runs `V*` once on a dummy transcript: random `root0`; after `V*` opens `r1` the round-1
challenges (hence the public `beta`, `rhs`) are known, and the dummy messages are drawn from the
**honest-observable distribution** -- `w0, h0` uniform, `v0` uniform on `{v|_H = beta}`, `q0` uniform on
`{sum_H q = rhs}` (red-team-zk-2 G1: with `v0, q0` merely uniform, a `V*` applying the public checks aborted on the
dummy run w.p. 1 and never on the real prover -- advantage 1, a simulator bug, fixed on lane b-zk-fix-2); receives the
openings `(r1, r2)` (or an abort, which it outputs), rewinds `V*` to after step 0, computes `r, rho, rc, cols`
exactly as the verifier does, samples the fake tableau (witness rows uniform of degree `< k`, `s_prox` uniform, `v`
uniform on `v|_H = beta`, `h` uniform, `q` uniform on `sum_H q = rhs`, `u_cb` uniform) and solves `s_lin, s_qa,
u_ca` by Lagrange interpolation at the `t <= t_pad` opened points plus uniform slack so every column test holds at
`cols`; `w := r . p' + s_prox`.  Commits, replays `V*`: by binding it opens the same coins or aborts (Goldreich-Kahan).
*Rewinds (red-team-zk-2 G2):* against a `V*` that aborts on the replay with probability `p`, the loop needs `1/(1-p)`
expected rewinds -- not a constant; `redteam/simulator.py` caps at `max_rewinds = 8` and outputs an abort transcript
on exhaustion, a simulation error of `(1-p) p^8` (up to `0.039` at `p = 0.9`), so the demonstrated simulator is
exact only for `V*` with bounded abort probability; the standard GK estimator of `p` makes it expected-polynomial
with no such error and is not implemented.  Models covered in `redteam/run.py`: honest `V`; adaptive `V*` that opens a slot to a
value depending on `root` / `(w, h)` (needs a collision: the prover aborts having sent only blinded messages, the
simulator outputs the same abort); statement-adaptive `V*` that grinds `r2` before step 0 against a target column
set (legal -- an interactive verifier owns its coins; the masking hides any `t <= t_pad` columns; simulated like any
coins); selectively aborting `V*` (abort on a predicate of `root`, or of `q`: the latter no longer correlates with
the witness after F1; or on the public checks `v|_H = beta`, `sum_H q = rhs` -- `zk2_dummy.PublicCheckVerifier`,
which must abort in neither world).  Indistinguishability: per-position chi-square real vs simulated on every message and on the
opened columns, plus the two `q` distinguishers of the red team (witness-free consistency, known-witness recovery),
which must now be at chance on real and simulated transcripts alike.

### Fiat-Shamir mode (`--mode fiat-shamir`) -- proof class COMPLETE_HVZK_BACKEND at most with `--zk`

No step 0.  `r, rho [, rc] = SHAKE(H("ligero-b/c1" || stmt || root))`, `cols = H("ligero-b/c2" || stmt || root ||
w h [q] [v])`; the transcript carries no coins; `rounds.sequential_depth = 1`.  This is the pre-8c protocol.

*Soundness.*  The campaign's rule (`note:r20-proof/zk-construction/20260922T0737Z-report-zk-construction` 3.3, `zk_cost.fiat_shamir_reparameterisation`): the generic
state-restoration bound `eps_FS <= q x eps_round`, `q = 2^60` hash queries.  `soundness()` multiplies every
statistical term by `2^60` and `config_for` re-sizes: per-proof target `-132.64 - 60 = -192.64`, **t = 288** (was
197; `t_pad = 512` so that Lemma 4.15 `k > l + t` holds: `k = l + 512`), and the field terms `(n + 3) / p^D x 2^60`
force **D = 7** (BabyBear^6 gives `2^-110` after the shift): union `2^-128.04` over 25 x 170.  Non-ZK FS: `t = 285`,
`D = 7`.  The 60-bit shift is the cost of removing the verifier; the interactive mode charges none of it.

*Hash assumption.*  The random-oracle model for the challenge hash -- for soundness (the FS transform) AND for zero
knowledge: the construction is HVZK (8b masking, Lemma 4.15) and a NIZK simulator must program the oracle at the
transcript it outputs (programmable ROM).  Malicious-verifier ZK in the CRH-only sense is not claimed here.  A
`bench-vu --mode fiat-shamir --zk` run is labelled **COMPLETE_HVZK_BACKEND** with the note "NIZK in PROM".

### What both modes share

*What a malicious verifier can no longer do (interactive).* Choose `r1` after seeing `root`, or `r2` after seeing
`w, h, q, v`: opening `c_i` to anything but the step-0 value needs `(r', s') != (r, s)` with `H(r'||s') = H(r||s)`.
Every coin the prover accepts is a fixed function of randomness sampled before the prover's first message; the
verifier's only remaining freedom is *whether* to open the next slot (abort) and which coins to commit to (any
coins give the honest-verifier view at those coins: the masking is coin-oblivious for every in-domain coin, and
`t <= t_pad`).  An equivocating verifier makes the prover abort having sent only `root` (slot 1) or
`root, w, h, q, v` (slot 2) -- blinded messages, never an opened column.

*Statement.*  `statement_digest(a, b, y16)`: the operands and the output words are PUBLIC in this statement, so the
witness is a function of the public input and privacy is vacuous for it today.  Every ZK label of this backend is
a statement about the construction (masking + coins + derivation) and becomes meaningful once the operands are
committed instead of published (auth-integration's LEAF_V2).  No private-operand statement is implemented here.

*Relations (`protocol.RelationHooks`; `fp8/`, `relations.py`).*  The same protocol proves other transition units:
the digest is `BLAKE2b("ligero-b/statement|v2" || rel.tag || "|p=..." ... "|sys=" || system_id || layout || "|a=" ||
a.astype(rel.word_dtype) || "|b=" || b.astype(rel.word_dtype) || "|y16=" || y16.astype(rel.y16_dtype) [|| auth])`.
The BF16 units (Ampere `REAL`, Hopper `Params.from_model(HOPPER_BF16_M16N8K16)`) use `tag = b""` and `<u2` words and
are told apart by `system_id` alone; the E4M3 units use `tag = b"|rel=fp8-ada|v1"` / `b"|rel=fp8-hopper|v1"`, `<u1`
operands (K = 32 per unit) and `<u4` chain-end words that are the FP32 accumulator with its ten always-zero low bits
dropped (`pack_public`, 22 bits, `y16_bits = 22`).  The public decode is the relation's (`unit.public_vectors`:
`t = 255` rejected; `fp8.relation.public_vectors_fp8`: the NaN codes `0x7F / 0xFF` rejected, `t = 15` finite).  The
file form is `ligero-statement/v4` (`serialize.py`: `u8 word_bytes | u8 y_bytes | str relation` before the words,
`u8 has_auth` after; a BF16 Ampere statement stays v2 / v3).  A file verifier must take the tag, the widths and the
word range from the relation the statement NAMES, never from the file, and must pin the relation's compiled system
by digest (`archive/ligero-verify/src/relation.rs`; `archive/ligero-verify/discrepancies/d11.md`, D11).

**Pole redraw.** B-Ligero has no retry-on-bad-challenge path: `Z_H` is invertible on the coset by construction, there
is no lookup helper with a public pole, and the column challenge is a deterministic rejection of repeats inside the
SHAKE stream (`_challenge2`: public data only, no prover-controlled restart, uniform over ordered `t`-subsets given a
uniform stream, `u mod n` exact for the power-of-two `n`). Nothing has to be redrawn under committed coins.

**Abort leakage.** The prover aborts only on (i) a coin opening that does not match its commitment or is malformed
(public data), and (ii) in the `check=True` debugging mode, its own constraint self-check -- which never fires for a
valid witness and is off in every measured run. No abort depends on a valid witness; the mask sampler cannot fail
(rejection top-up, not abort); hint generation raises only on off-domain public operands.

**Soundness accounting** (`soundness()`): interactive -- the commitment adds `commitment_hash` = hiding vs the
prover, `2 x q / 2^256` with `q = 2^64` = `2^-191` per proof, union-bounded with the rest; at 25 x 170: `t = 197`,
union `2^-128.05` (ZK) / `t = 196`, `2^-128.25` (non-ZK).  Fiat-Shamir -- every statistical term `x 2^60`; `t = 288`
(ZK, `t_pad = 512`) / `285` (non-ZK), `D = 7`, union `2^-128.04`.  The result JSON carries `mode`, `bound_model`,
`hash_assumption`, `zk_statement`, `fiat_shamir_queries_log2` per run.

**Cost.** Interactive: no arithmetic; 192 transcript bytes; four Blake2b calls; one more round trip; the F1 fix adds
one `D`-row mask family (`+D` committed rows) and one `D x (l-1)` mask block.  Fiat-Shamir: `t` 197 -> 288 opened
columns (+46% opened bytes and paths), `D` 6 -> 7 on every test, `t_pad` 512.  Measured numbers: `note:r20-proof/b-zk-fix/20260922T1142Z-report-b-zk-fix`.

**Mask sampler** (`mask_sampler.py`): ChaCha20 counter mode on the device under a fresh 32-byte key, reduced to
BabyBear by rejection (`u < 2p`, then `u mod p`): exactly uniform given the stream. The 8b sampler's direct
`uint32 mod p` had a statistical distance of ~0.11 per symbol (residues below `2^32 - 2p` are 1.5x as likely), so the
HVZK simulator was only close through that bias; now the simulator's distance is ChaCha20's PRF advantage. Kernel
checked against a pure-Python RFC 8439 block function and the RFC's test vector.

**Encoder** (`encode_simt.py`): the b-encode/b-v2 int32 Montgomery radix-4 SIMT kernel specialised to this code's
coset `g <w_n>` and natural column order, with the ZK mask folded in per coset (`Z_H` is the constant
`g^l w_n^{cl} - 1` on coset `c`, so `Z_H s_j` costs 256 multiply-adds per coset per row, no second transform) and the
natural-order coefficients of `interp_H(W_j)` emitted for the coefficient-form messages `w, v`. Bit-exact against
`field.encode` / `eval_on_coset` at `l = 256 .. 16384`. Per proof of 3516 rows: `l = 16384 -> 65536`: 145 -> 9.6 ms
(the 64 KiB shared-memory row allows one CTA per SM); `l = 8192 -> 32768`: 68 -> 2.1 ms (225 GB/s of codeword).

**Tests** (`tests_fused.py`): the D-fold combinations `r W`, `alpha p`, the booleanity sum `sum_j rho_j (U_j^2 - U_j)`
and the general quadratic constraints `sum_q rho_q (A_q B_q - C_q)(U)` are each one pass over the matrix (one thread
per column, the rows split across blocks, three 62-bit products per 64-bit reduction), reading the int32 codeword
directly and evaluating `A_q, B_q, C_q` from CSR tables without the `(Q, n)` intermediates. Exact mod p, bit-exact vs
the torch paths (`tests_fused_test.py`). The protocol is unchanged: the same polynomials, the same messages.

**Measured (README, third pass).** 4096 VUs on the L40S, 25 x 170: non-ZK 1.37 s (0.334 ms/VU), HVZK 1.43 s,
**ZK 1.41 s (0.345 ms/VU)**; verifier 0.96 s; 122.3 MB; depth 3; union 2^-128.05; ZK cost +3%.

### Batch verification: the union bound is a verifier check (lane b-batch-bound)

A batch of `B` VUs is proved as `N` **independently committed** sub-batch proofs (`bench-vu`: `N = ceil(B / B_proof)`,
25 x 170 at the decision workload).  Each sub-batch statement binds its own words and parameters (section 3/4: the
digest is computed by the verifier from its own configuration; nothing here weakens that) plus the count `n_proofs`
the prover sized `t` for.  Every per-proof bound `2^-b_i` (section 6, `soundness()`) is a statement about ONE
commitment; the only thing that makes "the batch is sound to `2^-t`" true is the union bound over the `N` proofs.
**A batch verifier with target `t` bits accepts iff**

1. every sub-batch proof verifies against its own statement, with the shared system (sections 3-8c as before);
2. `sum_i 2^-b_i <= 2^-t`, where `b_i` is the verifier's OWN recomputation of proof `i`'s bound from that proof's
   parameters `(n, k, l, t, D, mode, zk)` -- never a prover-supplied field -- and `t` is the verifier's parameter
   (`--target-bits`, default 128); the batch bits reported are `-log2 sum_i 2^-b_i`;
3. every proof's `n_proofs` equals `N`, the number of sub-batches presented (a proof sized for a smaller batch is
   rejected, reason `batch: n_proofs mismatch`; so is one sized for a larger batch -- it belongs to another batch).

Rule 2 is the accounting; rule 3 is what makes a sub-batch proof's `t` a commitment to its batch rather than a
coincidence.  Before this lane neither verifier enforced either: `N` proofs each claiming `n_proofs = 1` at
`2^-132.9` were accepted while the batch was only `2^-128.3`, and `N` proofs of a 25-batch presented as a 26-batch
would have passed with `2^-128.2` claimed as `2^-128`.  A single proof verified alone (`ligero-verify verify`,
`protocol.verify`) is not a batch verdict: it checks its transcript and its per-proof accounting union-bounded over
its OWN `n_proofs` (D1 caveat: the Python `verify()` checks the transcript only).  The reason strings are the same
in both verifiers (`batch: n_proofs mismatch (...)`, `batch: union bound 2^-x over N sub-batches exceeds the target
2^-t`); the accept reason carries the batch bits.  Soundness is a parameter, not a constant.

## 8d. `authentication = included-hash`: the operand rows hashed inside the relation (lane hash-relation)

`authentication = included` (D9) bound the published operand words to `verity.ml.commitments` roots by having the
*verifier* re-hash every word as a SHA-256 leaf: the words are public and the verifier pays 3.6 s per 4096 VUs.
`included-hash` keeps the words **private** (witness rows under the ZK masking of 8b) and moves the hashing into the
relation: every VU's x row and W column are hashed with **Poseidon2 over BabyBear** by constraints, the two digests
are public words at the chain end, and the verifier checks them against the roots with ordinary SHA-256 Merkle
paths.  Binding = Poseidon2 collision resistance (8 lanes = 248 bits of digest, 2^-124 birthday) + the SHA-256 ids
tree; privacy = the Ligero ZK rows + the one-wayness of the digest of a 1536-word row (a low-entropy row would need
a salt; not done).  Nothing in sections 1-8c changes: the composed relation is another `System` (`hashchain.compose`)
proved and verified by the same `protocol.prove / verify`; the bare relation and its files are byte-identical.

**The relation** (`hashchain.py`, composed over the chain-compiled bf16-hopper, fp8-hopper and fp8-ada units without
editing their compilers; bf16-ampere and fp4-nvf4 are not composed):

1. *Private operands.*  Every operand pin `a[i].s / .m / .e` (and `b[i].*`) of the unit becomes a hint row; the word
   behind it is `word_bits` boolean hint rows `hash.a[i].b<j>` and the decode is re-established by constraints on
   them (`s = b_msb`, `m = f + nz 2^frac`, `e = t - nz - (bias - 1)` with `nz = [t != 0]` as `t * inv = nz`,
   `(1 - nz) t = 0`, `nz` boolean, and the finiteness the verifier used to check on the public word as one
   `inverse_nonzero`).  The intervals the pins declared are theorems of these rows, so the unit compiler's audit
   stands.  18 rows per BF16 word, 13 per E4M3 byte.
2. *Two sponges along the chain.*  Poseidon2-BabyBear width 24, rate 16, capacity 8, S-box x^7, R_F = 8, R_P = 21,
   Plonky3's constants and permutation order (`verity.ml.commitments.poseidon2_babybear`; the constants, the internal
   diagonal and M4 are pinned by `CONSTANTS_SHA256 = 0b81af1c...`, known-answer vector `KAT_24` from Plonky3's
   `test_default_babybear_poseidon2_width_24`).  Padding-free overwrite-mode sponge: capacity initialised to
   `IV(role, word_bits, n_words) = (role, word_bits, K, 0, ...)` with `role = 1` for x rows and `2` for W columns
   (domain separation of the two tensors and of the row format), every block of 16 elements overwrites the rate
   part, one permutation per block, digest = the first 8 rate lanes of the final state.  An element carries 16 bits
   of consecutive words (`pack_words`: one BF16 word, two E4M3 bytes little-endian), so unit `i` absorbs exactly its
   16 lanes of `a` into sponge X and of `b` into sponge W: **two permutations per unit**, 213 S-boxes each as 4
   product rows (`x^2, x^3, x^4, x^7`) = 852 rows per permutation, the linear layers as affine expressions (no rows).
   The 8 capacity lanes of each sponge are *linked* rows across the chain (`sys.chain["c"] / ["y"]`: rows hold
   `cap - IV`, so a chain start is 0 and `cap_in[j + 1] = cap_out[j]`): 16 linked rows beyond the accumulator's
   three.  `chain.py` gives the extra links coefficients `u_(e mod 6) * u_6^(e div 6 + 1)` (monomials in the seven
   coefficient families instead of new families: `N_FAMILIES` is fixed by `protocol.py`); the chain identity is then
   a polynomial of degree <= 4 in the families instead of 1, so the field term of the chain test gains a factor <= 4
   on a 2^-17x-class term -- immaterial, and not in `soundness()`'s formula.  The three-link tables are unchanged, so
   every bare relation's proof is byte-identical.  System file `ligero-system/v3` (`LIGSYS03`) carries the variable
   link count.
3. *The digests at the chain end.*  Public pin `hash.is_end` (1 at a VU's last unit, 0 elsewhere; a function of
   the layout) and 16 public pins `hash.d[0..15]` (x digest lanes, then W digest lanes; 0 off the ends) tied by
   `is_end * (state_out[x] - d[x]) = 0`; the relation's own public word `y16` stays the chain end it was.  The
   public table is `(l, 17)` instead of `(l, 192 + 1)`: a v5 statement is ~6 kB where v4 carried 27 MB of words.

**Census** (rows per unit, bare -> hashed; per VU x 96 / 48 units): bf16-hopper 3292 -> 5637 (+608 private-operand
rows, +1720 hash rows, +17 pins; +71 %), fp8-hopper 3396 -> 5837 (+72 %), fp8-ada 3769 -> 6210 (+704 / +1720 / +17;
+65 %).  The measured cost (one RTX 4090, fp8-ada, 4096 VUs, l = 16384, 3 reps, bare vs committed on the same pod)
is in the lane note `hash-relation` and in the store (`authentication=included-hash` labels).  Prover implementation
note (no protocol content): the sponges' 1704 S-box rows per unit are written by a dedicated device kernel
(`hashchain.Poseidon2Witness`, a `witness_device` hook: the permutation as straight-line code with the 24-lane state
in registers) rather than by the fused witness program evaluating the linear layers as ~74k-term affine forms; the
witness is identical, so the proofs are.

**Committer** (`verity.ml.commitments.rowleaf`, `leaf/v2h`; `hashauth.py`): tree `a` = one leaf per x row (the row's
Poseidon2 digest as 32 big-endian bytes, schema `poseidon2-babybear-w24/row/v2h`), tree `b` = one leaf per W column,
tree `y` = the output words as in D9; the SHA-256 ids tree, framing, domains and multiproofs of `verity.ml.commitments`
are unchanged.  Gadget digest == committer digest on the same words (tested on random rows for all three relations).

**Statement v5** (`LIGSTM05`, `serialize.py`): the v4 header (K = words per committed row = steps x operands per
unit), `u8 y_bytes | str relation | str "included-hash" | n_vus x 16 digest lanes (<u4, canonical) | l y words |` the
block `str "ligero-b/auth/v2h" | str schema | 32 B CONSTANTS_SHA256 | u32 n_vus | n_vus x (u32 vu_index, u32 x_index,
u32 w_index) | 3 tree refs (D9 layout) | 3 x (u32 n_siblings | 32 B each)`.  The statement digest absorbs the block
(version, schema, parameters digest, index rows, tree refs) through the relation tag and the digests and y words in
the body, so a proof for other rows / roots / ranks is a different statement; the siblings are auxiliary (a wrong
path fails the path check, D9 order: Ligero tests first).  Rust `ligero-verify` parses v3 systems and v5 statements,
pins the three hashed systems apart from the bare ones (`relation.rs hashed_sys_id / hashed_table_digest`), and runs
the path check natively (`auth.rs check_hashed`; fixture `fixtures/fp8-ada-hash`).

**Sharing** (`--tile NXxNW`): a 64 x 64 GEMM tile is 4096 VUs over 64 x rows and 64 W columns; the trees have 64 + 64
leaves and the multiproofs open every leaf (~depth siblings per distinct row, 288-480 B per sub-batch).  With
`included-hash` every VU still hashes its own two rows; `included-hash-shared` (8e) hashes each distinct row once per
sub-batch.

* Merkle: `verity/ml/kernels/hash_gpu` Blake3 on the device (merged from main during the night; bit-exact with the
  host `blake3`, so `verify_path` is unchanged and the host path remains the CPU fallback). Before it, the transfer
  of `U` to the host (0.37 s) and host hashing (0.28 s) were 65% of the GPU-side prover; now 2 ms.
* Hints: `hints_torch.py` vectorises `gen_step` (20 ms per 4096 units vs 2.6 s in Python); the gate asserts equality
  with the checker's `gen_step` on every proved unit. A wrong hint is a completeness bug (the constraints reject it),
  never a soundness one. On the device as one CUDA graph (17 ms per 4096 units incl. the public decodes).
* Kernel fusion (final state): `torch.compile` on the NTT butterfly and on the constraint-table kernels (`_combine`,
  `apply_sparse`), and CUDA-graph replay of the whole witness program (~2500 launches -> one graph; 55 -> 8 ms).
  The hint generator is graph-replayed too. Prover 0.095 s per 4096 units (23.2 us/unit), 0.108 s per 42 VUs
  (2.57 ms/VU), 0.195 s per 85 VUs at `l = 8192` (2.30 ms/VU); `results/result_*.json`.
* The hash and the SHAKE-256 challenge expander remain "placeholders" in the sense of the shared brief (not the
  campaign's final choice of hash).
* ZK masking: section 8b, implemented by lane b-ligero2 (`--zk`); malicious-verifier ZK (HM96 coin commitment, coin-only challenges): section 8c, lanes b-zk / b-zk-fix (COMPLETE_ZK_BACKEND in the interactive mode only; the statement's operands are public, so the label is about the construction).
* Challenge expander: SHAKE-256 with `u mod p` on the *challenges* keeps the 2^-31 per-residue bias (a factor <= 1.5
  on the `1/p^D` field terms, 2^-172 and below: immaterial); only the mask sampler was moved to rejection.

## 8e. `authentication = included-hash-shared`: each committed row hashed once per sub-batch (lane share-logup)

`included-hash` (8d) hashes a row once per VU: at the 64 x 64 tile every x row is hashed 64 times and every W column
64 times (2 x 4096 sponges of 48 permutations per sub-batch of 341 VUs for ~70 distinct rows), and the hash rows
(1720 of 6210 per unit, +65 % over the bare relation) are the whole cost of the committed-operand relation.  Sharing
makes the hashing proportional to the **distinct** rows a sub-batch touches and leaves the GEMM proof to prove only
that every VU's operand words *equal* a shared row -- without re-hashing and without a lookup argument.

**Two mechanisms were censused.**  (a) *LogUp multiset*: a `(row_id, word)` lookup of every operand word of every VU
into a table of the distinct rows' words -- a rational-function sum per row with a post-commitment coin and a running
product / fraction row per VU (>= 3 rows per unit plus inverse hints, a batched inverse per column), the table proof
being another Ligero proof over the distinct rows.  (b) *Fingerprint chain*: a coin `rho` drawn after the root and one
linear constraint per VU, operand slot and coordinate saying "the rho-weighted sum of my 768 lane values is F"; the
same sum over the shared row is proved on the hash side.  (b) costs **no witness row** (linear in committed rows with
rho-dependent coefficients: it is more constraints of the general linear test the tests already run), so (b) was built.

**The proof of a sub-batch is a pair.**  `G` (`hashchain.compose_shared`, the GEMM side): the chain-compiled unit with
private operands as in 8d step 1 (704 rows for fp8-ada), the 16 lane rows per operand slot of 8d's sponge input
(`hash.a.lane<i>` / `hash.b.lane<i>` = the packed 16-bit elements a sponge would absorb, `word_bits` bits per word
re-established as in 8d) but **no sponge**, one public pin `hash.is_end`; 4506 rows per unit (bare 3769: +19.6 %).
`H` (`hashchain.row_hash_system`, the hash side): one unit per DISTINCT committed row of the sub-batch (the distinct
`x_index` values of the authentication block, then the distinct `w_index` values: `hashchain.shared_units`, a function
of the block alone), 48 columns per unit holding the row's 256 bits, 16 lane rows, one Poseidon2 permutation per
column (852 S-box rows), 8 linked capacity rows, and public pins `is_end`, `role` (0 = x row, 1 = W column: the sponge
IV is `IV_x + role (IV_w - IV_x)`, so one system hashes both tensors) and the 8 digest lanes at the unit's end; 1142
rows per unit, 70 units at the 64 x 64 tile -> `l_H = max(256, l_G / 4, next power of two >= 48 n_units)` (floored so
the sub-batches of one batch share a layout).

**The fingerprint constraints** (`chain.py` "Fingerprint constraints"; `sys.chain["fp"] = {lanes, coords C, mode}`).
Both systems carry, for slot `v` (a VU of G, a unit of H), operand `o` (G: x, W; H: the row) and coordinate `x < C`:

    fp_{v,o,x}:  sum_{s < 48} sum_{i < 16} rho_x^(16 s + i) * lane[o][i][48 v + s]  =  F[v][o][x]

as extra constraints of the chain test at slot `v`'s END column, with the end column's unused monomials as their
uniform coefficients (8d's `extra_coef` rule: extras `2 n_extra_links + o C + x`; the chain identity stays a polynomial
of degree <= 4 in the seven families).  The lane-row coefficient at column `48 v + s` is `sum_x u_{v,o,x} rho_x^(16 s + i)`
(one `(D, n_op, C, slots, 48, 16)` product per proof, on the device, graph-captured) and the right-hand side gains
`sum u_{v,o,x} F[v][o][x]`; a pad slot carries none.

*Coins and messages (8c holds).*  `rho_0..rho_{C-1}` are the tail of **challenge 1** of G -- `SHAKE-256(seed_1)` expands
`C` more elements after `r | rho_lin | rho_quad | rc` (`protocol._challenge1(n_fp=C)`), so `rho` is a function of the
verifier's coins alone (interactive: `r1`; Fiat-Shamir: the root) and is drawn AFTER the root that commits the lane
rows.  `F_G` (`n_vus x 2 x C` field elements, `Proof.fp`, `ligero-proof/v3`) is a **prover message** sent after coin 1
and absorbed into challenge 2 (`seed2_preimage(..., fp=)`: `|fp=` + F in Fiat-Shamir mode; in the interactive mode the
column challenge is a function of `r2` alone, so no absorption is needed and none is done).  `H` takes `rho` and
`F_H` as **statement values** (`statement_digest(fp=)` absorbs `|fp_rho=` rho `|fp_F=` F, so H's challenges are bound
to them), where `F_H[unit] = F_G[v][o]` for the VUs `v` whose operand `o` is that unit -- the verifier builds `F_H`
itself and REJECTS if two VUs of one unit sent different fingerprints.  H is proved second, on its own step-0 coins.

*What the verifier checks* (`relchain.SharedHashedRunner.verify_vus`; Rust `verify.rs verify_pair`): (1) `G` with its
own `rho` and the prover's `F_G` (the Ligero tests, the chain test with the fingerprint constraints, the public
`is_end` pins; `verify(out=)` hands `rho` out); (2) every unit's VUs agree on `F`; (3) `H` on the statement
`(table_H, units, rho, F_H)` -- the table's roles and `is_end` columns are the units' (`shared_units(auth)` recomputed),
its chain ends on the constant 0 and the digest pins are 0 off the ends; (4) H's digests, expanded per VU through the
`VU -> unit` maps, are `leaf/v2h` leaves at the block's ranks under the `a` / `b` roots and the y words under the `y`
root (8d's `verify_hash_auth`, unchanged).  A wrong `rho` on H (replaced, degenerate) is a different H statement and
fails H's tests; H's own coins are independent of G's.

**Soundness** (`protocol.fingerprint_collision`, booked in `soundness(fp=(C, steps, n_op))` of the GEMM side; the hash
side's `F` are statement values, so it books nothing new).  A VU whose lane words (the committed row G computes on)
differ from the shared row H hashed, yet passes, needs `F_G[v][o] = F_H[unit]` on every coordinate: per coordinate the
difference of the two fingerprints is a nonzero polynomial of degree < 16 x 48 = 768 in `rho_x`, and the
Schwartz-Zippel bound is `(16 steps - 1) x max_r Pr[rho_x = r]`.  The coins are `uint32 mod p` (`_expand`, no
rejection: a residue has at most 3 preimages of 2^32), so the maximum is `3 / 2^32` rather than `1 / p`; the `C`
coordinates are independent; the union is over the proof's `n_op x (l / steps)` fingerprinted slots (a VU's x row and its
W column are two events).  Per proof:

    fingerprint_collision = n_op (l / steps) ((16 steps - 1) 3 / 2^32)^C
      fp8-ada, l = 16384, 48 steps, 682 slots:  C = 8 (interactive)  2^-157.2      C = 10 (Fiat-Shamir, x 2^60)  2^-138.9

It is a statistical term (rho is drawn after both rows are committed), so it takes the Fiat-Shamir factor `2^60`
like the query terms; that is why the two modes carry different `C` (`chain.FP_COORDS = 8`, `FP_COORDS_FS = 10`:
with `C = 8` the Fiat-Shamir term would be 2^-110.6 > 2^-128).  The rest of the pair's bound: both sides are sized
for `2 n_proofs` commitments (`config_for(..., 2 n_proofs)`), the batch rule counts `2 N` entries (`.G`, `.H` per
sub-batch; `verify_batch_dir`, Rust `--system-h`), and the Poseidon2 binding is 8d's (the digest of the shared row is
what the trees commit).  What is NOT covered: a VU may equal a shared row *other than the one its index names* only
if the prover's `F_G[v][o]` equals that row's `F_H` -- the verifier's `VU -> unit` map is the authentication block's,
so this is the same event as above.

**Privacy (red-team-leaf F8).**  `F_G` publishes `C` verifier-known random linear functionals (~31 bits each: 248 /
310 bits) of EVERY private operand row a sub-batch touches -- x rows AND W columns -- beyond the digests of 8d, for
every leaf.  The proof's `zk_statement` says so (`relchain ZK_SHARED_NOTE`).  Removing it means folding H's units
into G's masked witness (one proof, `F` an internal wire never sent; "v2"): ~+1142 x 70 rows on G's `l = 16384`
layout = +5 % of G's cells, no second transcript, no `H` statement -- not built.

**Formats.**  `ligero-system/v4` (`LIGSYS04`): v3 + the fingerprint block after the chain (`u32 n_op | n_op x 16 lane
rows | u32 C | u8 mode 0 coin / 1 public`).  `ligero-proof/v3` (`LIGPRF03`): the v1 body + `n_vus x n_op x C`
fingerprints (G only; H is a v1 proof in `<stem>.hproof`, its coins in `<stem>.hcoins`).  `ligero-statement/v6`
(`LIGSTM06`): the v4/v5 header, `u8 y_bytes | str relation | str "included-hash-shared" | l y words |` the hash side
`u32 l_H n_H D_H t_H t_pad_H n_units C | n_units x (u32 role, 8 digest lanes <u4) |` the 8d authentication block; the
units themselves are DERIVED from the block (`shared_units`) and the file's roles must agree.  Rust
`ligero-verify verify|batch --system system.bin --system-h system_h.bin` verifies the pair (`relation.rs
shared_systems`: the pinned `(G, H)` digests per mode).

**Cost** (one RTX 4090, fp8-ada, 4096 VUs = 13 sub-batches of 341 at `l = 16384`, 64 x 64 tile, interactive; the lane
note has the tables): per sub-batch G ~25 ms + H ~6 ms + hints ~11 ms against `included-hash` ~43 + 10 ms and the bare
relation (`--pipeline 4`) ~12 ms; rows hashed per VU 2 -> 70 / 341 = 0.21; proof bytes G 5.2 MB + H 1.0 MB per
sub-batch against 6.5 MB; the pair verifies in ~0.1 s (Rust).

## 9. Negative-test expectations

* Every `vu-k1536-neg` instance, as a unit-level claim (the last unit claims the FP32 word `claimed_y16 << 16`, whose
  BF16 cast is the wrong claimed word; `reject`-class instances have off-domain operands): the verifier rejects,
  `wrong` ones at the linear test (the `y.*` pins), `reject` ones at the public-input domain check.
* Mutated witnesses: one row of every `(gadget scope, row class)` (243 of them) changed in one column before
  commitment: the verifier rejects (linear or quadratic test) -- an accepted one is a soundness bug.
* Wrong `y` for a unit: rejected at the linear test.
* Chain: every `vu-k1536-neg` as a VU claim (`wrong`: the chain test's sum check; `reject`: the operand domain
  check, or any proof of a different statement); a broken link, a nonzero `c_0`, a flipped claimed `y16`: the chain
  test; mutated rows (252): linear / quadratic / chain test.
