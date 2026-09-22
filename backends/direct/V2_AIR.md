# Checker v2 as a BabyBear LogUp AIR: the column layout (lane b-air-v2, 2026-09-22)

The relation is the frozen one (`checker/v2/`: the same accepted set as v1 and the reference, PROTOCOL.md 1).
This file fixes how checker v2's gadgets become rows of the LogUp AIR of PROTOCOL.md 14 over BabyBear
(p = 2^31 - 2^27 + 1), what changes against the Python gadgets and why each change is an *embedding* (the same
predicate over Z, realised so that no identity can wrap mod p), and what it costs.  Code: `src/program_v2.rs`,
`src/air_v2.rs`; z3 verdicts: `backends/numerical/reports/b_air_v2_z3.json` (`redteam/z3_air_v2.py`);
protocol text: PROTOCOL.md 15.

## 1. Lookups are tuples, not packed keys

checker v2 packs every multi-input table key linearly (`4 (E - cand) + s4`, `u R + 2 dt + s`, `l + 31 (E - floor)`)
because its `lookup` predicate has one key.  Over F_p a packed key binds only a *linear combination* of the
components (the `state_shift` finding, checker_min.md 3.2), so the halves had to be range-checked separately.
The LogUp argument of PROTOCOL.md 14 combines the whole tuple `(k_1, .., k_a, o_1, .., o_b, tag)` with powers
of `beta`: membership binds **every component** to a table column, i.e. each key component is bound to its
column's value set for free.  Every v2 lookup below is therefore a tuple lookup (`Ctx::lookup_tuple`) with the
components as separate tuple entries; no linear key, no auxiliary range check on a key component.  The
preprocessed matrix holds up to 8 tuple entries + tag (`N_COLS = 8`, was 5; the beta-compression term of
14.2 grows by `log2(8/5) = 0.7` bits).

## 2. Tables (preprocessed, committed once per proof; heights at REAL)

| table | tuple | rows | replaces (checker v2) |
|---|---|---|---|
| `T_OP` | (w; s, m, e) | 65 536 keys, 65 280 usable (non-finite words are tag-0 padding rows) | `T_OP` |
| `ALIGN4` | (d, s4; dsp) with `dsp = 2 min(d, 26) + [s4 in {1,2}]` | 780 x 4 = 3 120 | `ALIGN4` (dt, sp packed into one output; `isz` dropped, section 4) |
| `SHIFT_HI` / `SHIFT_LO` | (u_c, dsp; term), `term = +-floor(u_c 2^18 / 2^dt)` / `+-floor(u_c 2^10 / 2^dt)`, `u_c < 2^8` | 2 x 13 824 | `SHIFT` (3.5 M rows) |
| `SSHIFT_2/1/0` | (M_c, dsp; term) at scales 2^17, 2^9, 2^1, `M_c < 2^8` | 3 x 13 824 | `SSHIFT_HI/LO` (2 x 221 k) |
| `LEAD10` | (x; nz, l), `x < 2^10` | 1 024 | `LEAD` (32 768) |
| `LEADNORM` | (l, E; koff, A1, e_hat, z, ovf, isfloor) | 31 x 392 = 12 152 | `LEADNORM` (A -> A1 = 2^(10+k), clamped to 0 when k > 13) |
| `TNORM10` | (v_0, koff; t), `t = floor(v_0 2^k)` clamped to 0 when `>= 2^24` (unreachable rows) | 1 024 x 32 = 32 768 | `TNORM` (1 048 576) |
| `T16`, `T15`, `T8` | range tables | 65 536 + 32 768 + 256 | `T_range16` (pack/epilogue limbs) |
| total | | **282 280** | 5.2 M |

So the trace has at least 2^19 rows (the 4 096-VU trace is 2^19 anyway; option (a) of the brief -- a 2^22-row
preprocessed SHIFT -- would have forced a 2^22-row trace, 8x the rows, and was rejected on that alone; option (c),
an implicit table, is the v1 division form with hints and is what v2 replaced).  Option (b), the chunked shift,
is exact by the lemma: `floor((A + B) / 2^d) = floor(A / 2^d) + floor(B / 2^d)` whenever `A` is a multiple of
`2^c` and `0 <= B < 2^c` (for `d <= c` both are exact; for `d > c`, `floor(B/2^d) = 0` and `A mod 2^d + B <
2^d`).  Applied to `u 2^10 = u_hi 2^18 + u_lo 2^10` (c = 18) and to `M 2 = M_2 2^17 + M_1 2^9 + M_0 2^1` twice
(c = 17, then c = 9).  z3 checks it at REAL (section 6).

## 3. One row = one transition unit; columns

Per row (BabyBear base field), in allocation order of `program_v2::unit_v2`:

| group | columns | count | constraints / queries |
|---|---|---|---|
| state in | `st.s`, `st.e`, `st.M` | 3 | `first * st.s = 0`, `first * (st.e - floor) = 0`, `first * st.M = 0`; transition links (section 5) |
| operands | `x.a[i]`, `x.b[i]` | 32 | 32 `T_OP` queries (w; s, m, e) |
| operand decode | `a[i].s/m/e`, `b[i].s/m/e` (T_OP outputs) | 96 | -- |
| per group g (x2) | `g.max.E` (hint) | 1 | -- |
| | `g.max[i].dsp` (ALIGN4 output), i = 0 (state), 1..8 | 9 | 9 `ALIGN4` queries (E - cand_i, s4_i; dsp_i); `cand_0 = st.e`, `cand_i = a.e + b.e`; `s4_0 = st.s`, `s4_i = a.s + 2 b.s` |
| | `g.max.attained.p1`, `.p2` (materialised products) | 2 | `p1 = prod_{i<5} (E - cand_i)`, `p2 = p1 prod_{5<=i<9} (E - cand_i)`, `p2 (E - floor) = 0` (section 4) |
| | `g.pshift[i].u_hi/u_lo` (hints), `.t_hi/.t_lo` (outputs), i = 1..8 | 32 | `u_hi 2^8 + u_lo = m_a m_b`; 16 `SHIFT_*` queries |
| | `g.sshift.M2/M1/M0` (hints), `.t2/t1/t0` | 6 | `M2 2^16 + M1 2^8 + M0 = M_c`; 3 `SSHIFT_*` queries |
| | `g.sum.sgn` (hint); `g.sum.v2/v1/v0` (hints); `g.sum.lead[c].nz/.l` | 1 + 3 + 6 | `sgn` boolean; `v2 2^20 + v1 2^10 + v0 = S - 2 sgn S`, `S = t_acc + sum t_i` (all terms table outputs); 3 `LEAD10` queries |
| | `g.sum.l` (materialised) | 1 | `l = nz2 (20 + l2) + (1 - nz2)(nz1 (10 + l1) + (1 - nz1) l0)` |
| | `g.norm.koff/A1/e_hat/z/ovf/isfloor` (LEADNORM outputs), `g.norm.t` (TNORM10 output) | 7 | 1 `LEADNORM` query (l, E; ..), 1 `TNORM10` query (v0, koff; t); `M' = (v2 2^10 + v1) A1 + t` is an expression |
| | | **68 per group** | |
| non-last finiteness | -- | 0 | `(1 - last) * g.norm.ovf = 0` for both groups |
| pack (BabyBear limbs) | `pack.M_hi8`, `pack.M_lo16` (hints), `pack.any_ovf`, `pack.inf_sign`, `pack.y_hi`, `pack.y_lo` (materialised) | 6 | `M_hi8 2^16 + M_lo16 = M_1'`; T8, T16 queries; `y_hi = s 2^15 + fld 2^7 + M_hi8` with `fld = e_hat + 126 + z (e_min - floor)` (< 2^16, no wrap: section 6); infinity blend per limb |
| epilogue (`cast.*`) | `hi2`, `lsb`, `carry`, `rem` | 4 | v1's `epilogue` on the limbs (T16 hi, T16 lo, T15 hi2, T16 rem, T16 y16; booleans; two identities) |
| chain | `vu.step`, `vu.first`, `vu.last`, `vu.inv`, `vu.y16` | 5 | as PROTOCOL.md 12.1; `vu.out: last (y16 - claim) = 0` |
| multiplicity | `lookup.m` | 1 | |
| **main width** | | **~283** | |

Queries per row: 32 + 2 x (9 + 16 + 3 + 3 + 1 + 1) + 7 = **105**, two of them degree 2 (the state candidate's
`s4_0` on group 1 is `sgn_0 (1 - z_0)`), so 107 degree units -> 27 helpers x 4 base columns = 108, plus `g` and
`S`: **aux 116 at BabyBear^4, 145 at ^5**.  Preprocessed 9.  **Committed width ~408 (^4) / ~437 (^5)** against
863 / 977 (v1 + LogUp): 2.1x / 2.2x fewer.  Program constraints ~160 (v1: 286), max degree 5 unchanged.  The
exact numbers are `direct-unit chain-layout --v2` (PROTOCOL.md 15 records them).

## 4. What differs from checker v2, and why it is the same relation

1. **Tuple lookups** (section 1): an embedding of the same lookups; strictly stronger over F_p than the packed key.
2. **Chunked shifts** (section 2): the same function `+-floor(x 2^s / 2^dt)` computed as a sum of chunk shifts;
   exact over Z by the lemma; every chunk is bound by its table tuple, the split identity has both sides
   `< 2^26 < p`.  This is the brief's option (b).
3. **`attained` as a product.** checker v2: `sum_i isz_i + isfloor >= 1` with `isz_i = [d_i = 0]` ALIGN4 outputs
   (9 columns + a 4-bit range check per group).  Here: `prod_i (E - cand_i) * (E - floor) = 0` (two materialised
   degree-5 products, no output columns).  Over Z the two are the same predicate ("some candidate or the floor
   attains E"); over F_p each factor is a residue of an integer in `[0, 780)` (ALIGN4 binds `E - cand_i` to the
   `d` column; LEADNORM binds `E` to `[floor, last]`), so a product of non-zero factors is non-zero (p prime).
4. **Three 10-bit chunks of `v`** (checker v2: two 15-bit halves through a 32 768-row LEAD): the bit length is the
   priority combination of the chunk bit lengths (one materialised column), the normalised mantissa is
   `(v2 2^10 + v1) A1 + TNORM10(v0, k)` -- exact because `v2 2^20 + v1 2^10` is a multiple of `2^10 >= 2^-k` for
   every `k >= -7`; for `k >= 0` all three parts are exact products.  `A1 = 2^(10 + k)` is set to 0 for `k > 13`:
   `k > 13` forces `l <= 10` (k <= 24 - l), i.e. `v2 = v1 = 0`, so the clamped rows multiply zero.  All products
   stay below 2^24 on every table-consistent row (z3, section 6): this is the "TNORM_HI"/`_v2_norm_lookup` 31-bit
   repair of checker_min.md 5, realised.
5. **Pack on limbs** (the F4 word class; checker_min.md 3.2: `epilogue`/`state_decode` SAT on 31-bit fields as
   stated).  The FP32 word `s 2^31 + fld 2^23 + M` (33 bits of range) never exists as one field element: `M` is
   split into `M_hi8 2^16 + M_lo16` (both sides `< 2^24`), the high limb is the *expression* `s 2^15 + fld 2^7 +
   M_hi8` (`fld <= 254` on a finite state, `M_hi8 < 2^8`, so `< 2^16`: no hint, no wrap), the low limb is
   `M_lo16`.  The infinity blend and the epilogue run per limb as in PROTOCOL.md 13 (LIMB16).  `state_decode` is
   absent from the state-chained VU (unit 0 starts from the constant zero state); its limb form for a word-in
   unit is the same split read backwards (`c_hi = s 2^15 + t 2^7 + f_hi7`, `c_lo = f_lo16`) and is listed in the
   z3 report for completeness only.
6. **State chained as `(s, e_hat, M)`**; `z` is a per-row LEADNORM output consumed by Pack on the same row
   (checker v2 chains `z` too, but no gadget reads the incoming `z`).

Nothing else changes: T_OP, ALIGN4 (minus `isz`), LEADNORM's function, the signed sum, `ovf = 0` on non-final
units, one Pack + epilogue per VU bound on the last row.

## 5. The chain

Rows `96 i .. 96 i + 95` are VU `i`; padding rows are the zero VU (PROTOCOL.md 12.1 unchanged: `vu.step`,
`vu.first`, `vu.last`, `vu.inv`, `vu.y16`, `next.first = last`, step increment).  The accumulator link carries the
**state**, not the word: on a transition row with `next.first = 0`,

    next.st.s = sgn_1 (1 - z_1),   next.st.e = e_hat_1,   next.st.M = (v2_1 2^10 + v1_1) A1_1 + t_1

(right-hand sides are the local row's group-1 output expressions, degree <= 2).  The first row of a VU has
`first (st.s) = first (st.e - floor) = first (st.M) = 0` (the constant zero state; unit 0 of checker v2 is
`static`: here its split hints exist and are forced to 0 by the identity).  Every non-last row asserts both
groups' `ovf = 0` (checker v2's `u{i}.g{g}.finite`).

Statement binding (red-team F1-F3, `lane/soundness-fixes` carried over from the start): the header (domain tag,
AIR id = hash of field + layout + constraint names, `B`, `K`, 96, `degree_bits`, FRI parameters, every claimed
word) is absorbed before the first challenge; the verifier fixes `degree_bits` from `B` and rejects a proof
whose height differs; the claim column `vu.y16` is checked at `zeta` and `zeta g` against the verifier's own
statement polynomial (barycentric evaluation, `statement.rs`).

## 6. z3 obligations (section 2 of the brief; results in `b_air_v2_z3.json`)

Per AIR gadget, at REAL widths, hints symbolic, tables symbolic (`Table.sym`): (i) over Z, "accepted and output
differs from the reference" UNSAT; (ii) over F_p (BabyBear), the wrap-free query -- every congruence holds with
multiplier 0 -- UNSAT, so (i) transfers; (iii) direct modular query over BabyBear where it finishes.

Gadgets: `product_shift_chunked`, `state_shift_chunked`, `signed_sum_chunked` (sign pinning needs `|S| <= S_MAX
< p - 2^30`, the V_PREMISE bound: checked as a premise), `normalise_chunked`, `group_max_product` (the product
form of `attained`; and SAT without it), `pack_limbs` (+ the epilogue on limbs), `state_decode_limbs`.  Tuple
lookups are modelled as one congruence per component against a symbolic row.
