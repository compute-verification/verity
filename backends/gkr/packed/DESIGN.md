# Packed (base-field-first) sumcheck for Candidate A's GKR c-rounds -- design

Lane `a-packed`, track A.  Question: how much of A's dominant bucket -- the checker sumcheck, 5.4e9 limb MACs per VU
in `backends/numerical/reports/tensor.md`, priced with a 24-limb extension challenge on both sides of every product
(576 limb MACs) -- does keeping the tables in the base field for the first rounds remove, soundly?

Everything here changes the **prover's algorithm only**.  The protocol (round polynomials, degrees, verifier checks,
soundness term) is untouched; the reference (`reference.py`) proves it by emitting bit-identical transcripts from the
naive and the packed prover, and the kernel (`sumcheck_packed.py`) emits the same transcripts on the H100.

## 1. The sumcheck (PROTOCOL.md 4.1, `explore.tensor` pricing)

One c-round of layer `i` is, over the batch of `units = 393,216` transition units and the layer's read width
(`2^10` for the computed layers, `2^9` for depth 1: tables of `units x 2^read`, i.e. `t = 2^28.6` elements, six
sumchecks per batch),

~~~text
claim = sum_{c in {0,1}^n} eq(gamma, c) . g(c) . h(c)          gamma in F_{p^6}^n,   g, h : {0,1}^n -> F_p
~~~

with `g, h` **base-field** layer tables (BabyBear `p = 2^31 - 2^27 + 1`) and the eq point `gamma` in `F_{p^6}`
(`X^6 - 31`, security target `2^-128` per `field_price_list`).  Round `i` sends `S_i(0..3)` (degree 3 in the round
variable: `eq . g . h`), the verifier checks `S_i(0) + S_i(1) = S_{i-1}(r_{i-1})` and, at the end,
`S_n(r_n) = eq(gamma, r) g(r) h(r)`.

`tensor.sumcheck_macs(n, 3, 24)` prices this as three *folded* tables, all in `F_{p^6}` after round 1: `8` triple
inner products of `dl^2 = 576` MACs plus three `2 dl x 2 dl` folds per pair -- `576 . 32 = 18,432` limb MACs per
element (summed over rounds), the 5.4e9 per VU.

## 2. Literature (45 minutes)

* **Gruen, "Some improvements for the PIOP for ZeroCheck", ePrint 2024/108.**  Two ideas.  (a) *The eq factor is
  not a table*: `eq(gamma, r_{<i}, x, b) = eq(gamma_{<i-1}, r_{<i}) . eq(gamma_{i-1}, x) . eq(gamma_{>=i}, b)`, so the
  round polynomial is a known degree-1 factor times a degree-`d-1` polynomial `Q_i` that needs one fewer evaluation
  and never folds an eq table.  (b) *Univariate skip*: replace the first `s` rounds by one round whose polynomial has
  degree `d . (2^s - 1)`-ish over a coset, which changes the round-polynomial degree and the soundness term.
* **Dao-Thaler, "More optimizations to sum-check proving", ePrint 2024/1210.**  The eq split above in general form
  (section 3: `eq` as a tensor product of a prefix, the round variable and a suffix), plus the observation that when
  the tables are small values the first rounds' products are cheap; algorithms that trade extension-field
  multiplications for base-field ones round by round.
* **Bagad, Domb, Thaler, "Speeding Up Sum-Check Proving", ePrint 2025/1117.**  The small-value / base-field-first
  algorithm: for the first `k` rounds do not fold the tables at all; expand `g_i(x, b) = sum_a eq(r_{<i}, a) g(a, x, b)`
  so that everything the challenge touches is a *sum over `2^(i-1)` base-field partial results*, computed as
  base x base products, and only combine with the (extension-field) `eq(r, a)` coefficients at the very end of the
  round.  Cost exponential in `k` but in base ops; the paper's sweet spot is `k ~ 3..5`, after which one fold by all
  `k` challenges at once puts the tables in the extension field and the ordinary algorithm continues.
* **TensorZKP** (the limb layout we use, chunking bound `N < 33,025` for unsigned limbs; ours are signed balanced
  limbs so products are `< 2^14` and INT32 holds `< 2^17` of them).

## 3. The algorithm implemented (`reference.packed_prover`, `sumcheck_packed.Kernel`)

**Eq-trick (all rounds).**  `S_i(x) = eq(gamma_{<i-1}, r_{<i}) . eq(gamma_{i-1}, x) . Q_i(x)`, with
`Q_i(x) = sum_b eq(gamma_{>=i}, b) g_i(x, b) h_i(x, b)` of degree 2, computed at `x in {0, 1, 2}`; `Q_i(3)` by
extrapolation; `S_i(0..3)` from `Q_i(0..3)`.  The weights `eq(gamma_{>=i}, b)` are a suffix of the eq table, and because
`eq` is a tensor product they are split `eq(gamma_{i..i+kappa-1}, b1) . eq(gamma_{i+kappa..}, b2)` with `|b1| = K1 = 2^13`:
the weighted inner product is ONE matmul over `b1` (`M = rows . 4 limbs, K = K1, N = 24`) and a short extension-field
contraction over `b2`.  Nothing is ever folded for `eq`.

**Packed rounds `i <= k`.**  `g, h` untouched (base, int32).  `g_i(x, b) = sum_a eq(r_{<i}, a) g(a, x, b)`, so

~~~text
Q_i(x) = sum_{a, a' in {0,1}^(i-1)} eq(r_{<i}, a) eq(r_{<i}, a') . T_i[x, a, a']
T_i[x, a, a'] = sum_b eq(gamma_{>=i}, b) . g(a, x, b) . h(a', x, b)          (base x base Hadamard, base x ext weights)
~~~

`3 . 4^(i-1)` Hadamards per surviving pair (`g(a, 2, b) = 2 g(a, 1, b) - g(a, 0, b)` stays base), each one INT64
multiply and one reduction mod p on SIMT, 4 limbs, and a `4 x 24 = 96`-MAC weighted inner product on tensor cores.

**One fold after round `k`.**  `g'(b) = sum_a eq(r_1..r_k, a) g(a, b)`: base x extension, `2^k . 6` products per
output element (`192` limb MACs per input element as a matmul; the Triton path does it on SIMT).

**Extension rounds `i > k`.**  Ordinary: 3 Hadamards in `F_{p^6}` per pair (36 coefficient products), `24 x 24 = 576`
MAC weighted inner products, fold of both tables by `r_i` (`[x_e | x_o] x [C_(1-r); C_r]`, `48 x 48` per pair, TensorZKP
eq. 14 with `X^6 = W` inside the matrix).

`k = 0` is the eq-trick alone (the plain algorithm minus the eq fold); the verifier is identical for every `k`.

## 4. Cost model (limb MACs per element of the layer table, summed over rounds)

`explore.tensor.packed_sumcheck_macs(n, packed_rounds=k)`; `bench.mac_model` counts the same MACs as issued.

| k | packed rounds | fold_k | ext rounds | total | vs plain model 18,432 | base Hadamards / elem | ext Hadamards / elem |
|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 6,336 | 6,336 | 2.9x | 0 | 3.0 |
| 1 | 144 | 384 | 3,168 | 3,696 | 5.0x | 1.5 | 1.5 |
| 2 | 432 | 384 | 1,584 | 2,400 | 7.7x | 4.5 | 0.75 |
| 3 | 1,008 | 384 | 792 | 2,184 | **8.4x** | 10.5 | 0.375 |
| 4 | 2,160 | 384 | 396 | 2,940 | 6.3x | 22.5 | 0.19 |
| 5 | 4,464 | 384 | 198 | 5,046 | 3.7x | 46.5 | 0.09 |

Packed round `i`: `3 . 4^(i-1) . (t / 2^i) . 96`; fold: `2 . 192 t`; extension round `i`: `(t / 2^i)(3 . 576 + 2 . 2304)`.
The optimum is `k = 3` (the `4^(i-1)` Hadamard blow-up overtakes the halving of the surviving table at `k = 4`);
predicted saving on the 5.4e9 bucket: 2.9x from the eq-trick alone, 8.4x with three packed rounds -- in MACs.  Measured
(README.md, r20260922-062108-4b25): `k = 3` over `k = 0` is **3.8x** (the MAC model says 2.9x for that pair; the base x
base Hadamards of the packed rounds are cheaper per MAC than the `F_{p^6}` Hadamards and folds they replace), and the
kernels run at 1.4-1.6% of the INT8 peak -- SIMT modular arithmetic and HBM, not tensor cores, bound them, so the MAC
count is the wrong unit for the last factor of ~10.

## 5. Verifier-side change

None.  Same `S_i(0..3)`, same degree 3, same checks, same final evaluation `eq(gamma, r) g(r) h(r)`.  The test
`test_packed_matches_naive_and_verifies` runs `verify` on the packed transcript and `test_verifier_rejects_tampered_round`
checks the verifier still rejects.

## 6. Soundness

The prover's messages are a deterministic function of the same claim and the same coins, identical to the naive
prover's (bit for bit; `reference.py` and the kernel tests), so the interactive protocol is literally the same one:
soundness `n . d / |F_{p^6}|` per sumcheck with `d = 3`, `security.accounting`'s term for A unchanged, `2^-128` target
kept.  No new assumption, no new relation, no change to what is committed.  The only thing that changed is which
field the prover's intermediate products live in; the packed rounds are exact (`T_i` is a sum of base products,
reduced mod p, combined with extension coefficients in the field).

**Not done: univariate skip.**  Gruen's skip (one round of degree `3 . (2^s - 1)` over a size-`2^s` coset for the first
`s` variables) changes the round-polynomial degree and the soundness term (`(d 2^s) / |F|` instead of `s . d / |F|`) and
the verifier; it would need a `security.accounting` recomputation and a flagged protocol change.  Left as the next step.

## 7. INT32 / limb bounds

Signed balanced limbs `l in [-128, 128)`: a limb product is `< 2^14`, so an INT32 accumulator holds `< 2^17` products.
The inner-product depth `K1 = 2^13 <= 32768` (TensorZKP's chunk); the `b2` contraction finishes the sum in the field
in INT64.  Fold matrices: an output entry sums at most `2 . 6 = 12` (ext fold) or `2^k` (fold_k) limb products.  Triton
kernels: products of two values `< 2^31` in INT64, reduced before accumulation; Triton's `%` truncates toward zero so
residues are lifted to `[0, p)` before the limb split.

## 8. Layout shared with lane `tc-sumcheck`

Same field constants (`p`, `X^6 - 31`), same balanced signed limbs, same `[x_e | x_o] x [C_(1-r); C_r]` fold with
48 output columns, same chunk.  Their kernel is the plain (three folded tables) algorithm; ours at `k = 0` differs
only by the eq-trick (weights as a tensor product instead of a folded table), so the two `k = 0` numbers bracket the
eq-trick's own value on the two GPUs.
