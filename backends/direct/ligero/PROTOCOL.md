# Ligero-style direct-constraint argument for the transition unit (lane b-ligero, track B)

Status: NON_ZK_PROOF_DIAGNOSTIC. Implemented in this directory (Python/torch); measured on `vy-g3` (NVIDIA L40S).
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

## 8. Not done / labelled

* INT8 matmul DFT encoder (tensor.md's proposal): not measured; the int64 NTT was fast enough that encoding is not
  the bottleneck at this shape (README table). The matmul variant would matter once hashing moves to the GPU.
* Merkle on the GPU: placeholder host Blake3 (`backends/shared/hash_gpu` not on main at the time of writing).
* The chain (VU) and ZK masking: sections in README.md (design + estimate, labelled).

## 9. Negative-test expectations

* Every `vu-k1536-neg` instance, as a unit-level claim (the last unit claims the FP32 word `claimed_y16 << 16`, whose
  BF16 cast is the wrong claimed word; `reject`-class instances have off-domain operands): the verifier rejects,
  `wrong` ones at the linear test (the `y.*` pins), `reject` ones at the public-input domain check.
* Mutated witnesses: one row of every `(gadget scope, row class)` (243 of them) changed in one column before
  commitment: the verifier rejects (linear or quadratic test) -- an accepted one is a soundness bug.
* Wrong `y` for a unit: rejected at the linear test.
