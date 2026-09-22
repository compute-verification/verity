# Ligero-style direct-constraint argument for the transition unit (lane b-ligero, track B)

Status: NON_ZK_PROOF_DIAGNOSTIC without `--zk`, COMPLETE_HVZK_BACKEND with it (section 8b; lane b-ligero2). Implemented in this directory (Python/torch); measured on `vy-g3` (NVIDIA L40S).
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

`p = 15 * 2^27 + 1` (31 bits, 2-adicity 27). Reasons: tensor.md's 1.6x for B from 4-limb base products, and the
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
evaluations on the coset `g <w_n>` with `g = 31` (`n = 4 l`, rate 1/4 as in tensor.md's B row). `H` and the coset are
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
`validate()` needs `l < k`, so `k = l + 1` stands in, conservative) and `F_challenge = BabyBear^D`:

    l = 4096, n = 16384, t = 189, D = 6:
      irs_query (1 - e/n)^t = (5/8)^189 = 2^-128.1    linear_query ((k+l)/n)^t = 2^-189    quadratic_query (2k/n)^t = 2^-189
      irs_field (n+1)/p^6 = 2^-172      linear_field, quadratic_field 1/p^6 = 2^-186
      total 2^-128.1
    2^-80 sensitivity point: t = 118, D = 4 (fields 2^-110..-124).

`t` is `ligero_queries_for(target, n, l, l)`. The linear test sends nothing, so its query term is at most the Appendix
C one used. The proximity-gap statement for base-field rows combined by base-field coefficients is Ligero's own
setting (no subfield/extension assumption is needed here, since the combinations are base-field).

## 7. Communication (l = 4096, m = 3372, t = 189, D = 6)

    root 32 B; w: D l * 4 = 98 KB; h: D (l - 1) * 4 = 98 KB; cols 756 B;
    opened columns: m t * 4 = 2.55 MB; Merkle paths: t * 14 * 32 = 85 KB      total ~2.8 MB per 4096 units (~700 B/unit)

The opened columns dominate (Ligero's `t * m`); `m` is what a cheaper range-check realisation would shrink.
Chain (m = 3516, + `q`: D (2l - 1) * 4 = 197 KB): 3.14 MB per 42 VUs (75 KB/VU).

## 7b. The chain: the verification unit (`chain.py`, `vu.py`; `compile_unit(chain=True)`)

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

Gate (vy-g3, `results/gate_vu.json`): 64 `vu-k1536` VUs with the recorded accumulators (two proofs, 42 + 22)
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
  `s_prox` (degree `< k`), `s_lin` (`< t_pad`), `s_qa` (`< k`), `s_qb` (`< t_pad - 1`), `s_chain` (`< k - 1`).
* Messages: proximity `w_d = r_d . p + s_prox_d` (`k` coefficients, uniform; `r_d` is zero on the prox rows, which
  enter with coefficient 1); linear `v_d = alpha_d . p + Z_H s_lin_d` (`k` coefficients; the verifier checks
  `v_d|_H = beta_d` -- the public right-hand side, folded mod `x^l - 1` and NTT'd -- and `v_d(eta_c)` against the
  opened rows; uniform given `beta`); quadratic `h_d = P_d / Z_H + s_qa_d + x^k s_qb_d` (`2k - 1 - l` coefficients,
  every one masked); chain `q_d = Q_d + Z_H s_chain_d` (`l + k - 1` coefficients; `sum_H q = l (q_0 + q_l + q_2l)`).
* Statement digest absorbs `zk` and `k`; the verifier's shape checks use `k`, `2k - 1 - l`, `l + k - 1`, `M = m + 5D`.
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

## 8c. Malicious-verifier ZK: the step-0 coin commitment (lane b-zk; proof class COMPLETE_ZK_BACKEND)

**Transcript (interactive semantics; `protocol.py` docstring).**

~~~text
 0   V -> P   c1 = H(tag || r1 || s1),  c2 = H(tag || r2 || s2)     r_i, s_i 32 uniform bytes each; H = Blake2b-256
 1   P -> V   root
 2   V -> P   (r1, s1)                     P checks c1; abort (CoinOpeningError, public) on mismatch
              r, rho_lin, rho_quad [, rc]  = SHAKE-256( H("ligero-b/c1" || stmt || c1 c2 || root || r1) )
 3   P -> V   w, h [, q] [, v]
 4   V -> P   (r2, s2)                     P checks c2; abort on mismatch
              cols                         = H("ligero-b/c2" || stmt || c1 c2 || root || w h [q] [v] || r2)  (t distinct)
 5   P -> V   U[:, cols], Merkle paths
~~~

Three verifier messages on the critical path (step 0 can travel with the statement, but it is a message the prover
must wait for): **`rounds.sequential_depth = 3`** (was 2). The non-interactive reading (what `bench-vu` measures):
the prover plays step 0 itself (`Coins.sample()`), the transcript carries `c1, c2, r1, s1, r2, s2` (192 B), and the
verifier recomputes `c_i = H(r_i || s_i)` -- `verify(..., coins=own)` is the interactive verifier, which additionally
requires the openings to be its own randomness (a proof from another session is rejected).

**What a malicious verifier can no longer do.** Choose `r1` after seeing `root`, or `r2` after seeing `w, h, q, v`:
to open `c_i` to anything but the value fixed at step 0 it needs `(r', s') != (r, s)` with `H(r'||s') = H(r||s)`, a
collision of `H` (generic `q^2 / 2^256 = 2^-128` at `q = 2^64`; the `coin_commitment_binding_collision` privacy
term). Every coin the prover accepts is therefore a fixed function of randomness sampled before the prover's first
message; the verifier's only remaining freedom is *whether* to open the next slot (abort). Nothing about the
distribution of the coins is required for privacy -- `redteam_zk.py` commits to all-zero coins and to coins ground
against a target column set: the view is the honest-verifier view at those coins (real vs simulated `p > 0.03` on
every message, opened columns uniform, target-set overlap at chance), which is what the 8b masking already gives for
every in-domain coin. An equivocating verifier (`r1 := f(root)` or `r2 := f(w, h)`) makes the prover abort having
sent only `root` (slot 1) or `root, w, h, q, v` (slot 2) -- blinded messages, never an opened column.

**What the simulator needs.** The coins are fixed after step 0, so `S` learns them by running `V*` once on a dummy
transcript (random `root`, uniform `w, h, q, v` consistent with the public `beta` and chain right-hand side), receiving
the openings (or an abort, which it outputs). It then **rewinds `V*` to the state after step 0** and, knowing
`(r1, r2)` and hence `r, rho, cols`, prepares the columns it will open (uniform, `t <= t_pad`), the coefficient rows
whose values at `cols` match them (Lemma 4.15: `s -> (Z_H(eta_c) s(eta_c))_c` is onto), the Merkle tree over them, and
the messages; by binding `V*` opens the same coins or aborts, and the expected number of rewinds is constant
(Goldreich-Kahan). The unopened columns are hidden by the hash of a uniformly random column (random-oracle model);
statistically hiding leaves (HM96 salted leaves, `backends/gkr/PROTOCOL.md` 5.5) are not implemented.

**Pole redraw.** B-Ligero has no retry-on-bad-challenge path: `Z_H` is invertible on the coset by construction, there
is no lookup helper with a public pole, and the column challenge is a deterministic rejection of repeats inside the
SHAKE stream (`_challenge2`: public data only, no prover-controlled restart, uniform over ordered `t`-subsets given a
uniform stream, `u mod n` exact for the power-of-two `n`). Nothing has to be redrawn under committed coins.

**Abort leakage.** The prover aborts only on (i) a coin opening that does not match its commitment or is malformed
(public data), and (ii) in the `check=True` debugging mode, its own constraint self-check -- which never fires for a
valid witness and is off in every measured run. No abort depends on a valid witness; the mask sampler cannot fail
(rejection top-up, not abort); hint generation raises only on off-domain public operands.

**Soundness accounting** (`soundness()`): the commitment adds `commitment_hash` = hiding vs the prover, `2 x q / 2^256`
with `q = 2^64` = `2^-191` per proof, union-bounded with the rest. At 25 x 170: `t = 197`, union `2^-128.05` (ZK) /
`t = 196`, `2^-128.25` (non-ZK) -- unchanged to two decimals.

**Cost.** No arithmetic; 192 transcript bytes; four Blake2b calls; one more round trip.

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

## 9. Not done / labelled

* Merkle: `backends/shared/hash_gpu` Blake3 on the device (merged from main during the night; bit-exact with the
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
* ZK masking: section 8b, implemented by lane b-ligero2 (`--zk`, COMPLETE_HVZK_BACKEND); malicious-verifier ZK (HM96 coin commitment): section 8c, lane b-zk (COMPLETE_ZK_BACKEND).
* Challenge expander: SHAKE-256 with `u mod p` on the *challenges* keeps the 2^-31 per-residue bias (a factor <= 1.5
  on the `1/p^D` field terms, 2^-172 and below: immaterial); only the mask sampler was moved to rejection.

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
