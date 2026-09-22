# B-Ligero at the checker-v2 relation: the witness shape (b-v2 lane, 2026-09-22)

**Scope: the Ligero layout** of Candidate B for checker v2 (`backends/numerical/reports/checker_min.md`, `checker/v2/`:
the same relation as v1 -- every accepted pair is accepted by both -- with 33 mul / 9 lin / 89 lookups / 15 hints per
unit instead of 358 / 137 / 306 / 225, at the price of 5.2M fixed-table rows).  Computed by `v2_shape.py` from the v2
census under `VERITY_CHECKER_VARIANT=v2` plus the BabyBear repairs of `explore.fields.small_field_census`
(`v2_shape.json`); the measured buckets at this shape are in `backends/numerical/reports/b_measured_4090_v2.md`.
Nothing of the relation is changed here: the predicates are checker-min's verified gadgets; this file only decides where
their wires, hints, helpers and tables sit in the committed matrix and counts what the Ligero tests touch.

## 1. Layout decision: rows are AIR columns, aligned by unit (the transposed layout of `LINTEST.md` 2.3)

Row `(column j, block b)` holds slot `j` of the `l = 3840` consecutive units of block `b` (`blocks = ceil(393,216 / 3840) =
103`, 0.6% waste), `k - l = 256` randomizers per row (HVZK), coset-major systematic RS `[16384, 4096]` over BabyBear as
before.  Consequences, all of which v1's "rows hold whole units" layout lacked:

* a quadratic triple `x o y = z` whose operands are AIR columns is three aligned rows per block: **no operand copies**
  (v1's layout charged 3 slots per product and up to 716 alignment constraints per unit; the census' `columns_committed_B`
  estimate of 3 x mul + other columns becomes `columns_all`);
* the lookup helper identity `h_j (z_T - w_j(beta_T)) = 1` is one triple OVER `F_c` with `x = h_j` (6 committed coordinate
  rows per block) and `y = z_T ONE - sum_c beta^c W_{j,c}`, `z = ONE` **virtual** (linear in committed rows with public
  coefficients: the verifier evaluates them at the opened columns from the opened witness rows; `ONE` is the interpolant
  of ones on the payload, zeros on the randomizers) -- so a lookup costs exactly 6 committed slots and 0 linear constraints
  beyond the per-table sum;
* `r^T A` on every witness row is `pi[j] rho^(b l C) (rho^(u C))_u`: one fixed geometric row `G` for all of them (`M` below).

## 2. Per unit (F_t slots; census v2 at REAL widths + BabyBear repairs)

| class | v2 census | + BabyBear repairs | slots | note |
|---|---|---|---|---|
| inputs (32 BF16 words, chained `(s, e, M, z)` state) | 34 | | | |
| hints | 15 | | | v1: 225 |
| lookup outputs | 195 | | | bound by the lookup argument, never range-checked |
| products (`z` of the `mul` triples that are wires) | 30 | | | |
| materialised linear expressions | 45 | | | |
| **witness slots** | 319 | +12 (F4 limb hints / inputs / range columns, sign-tightening range columns: `fields.py` deltas, summed) | **331** | tableau 1 (round 1) |
| lookups | 89 | +6 (F4 limb range chunks 4, sign tightening, `TNORM_HI` 2 -- `fields.py` deltas; `T_range16` 10 -> 14) | **95** | `T_OP 32, ALIGN4 18, SHIFT 16, T_range16 14, LEAD 4, SSHIFT_HI/LO 2+2, LEADNORM 2, TNORM 2, TNORM_HI 2, T_HDR 1` |
| **helper slots** `h_j` (6 coordinates each) | | | **570** | tableau 2 (round 2) |
| **committed per unit** | | | **901** | v1 layout estimate: 1,112 + 6 x 320 helper coordinates = 3,032 |
| `F_t` quadratic triples (`mul`) | 33 | +1 (F4 3, `TNORM_HI` -2) | 34 | v1: 358 (327 at BabyBear) |
| `F_c` quadratic triples (helper identities) | 89 | | 95 | v1: 320 (as `F_c x F_t`: 6 coordinate products each) |
| `F_t`-linear identities (incl. range-chunk splits) | 9 | +2 | 11 | + 4 transition-linkage constraints per unit; v1: 147 |

The BabyBear repairs are checker-min's *modelled* deltas (`checker_min.md` 3.2, `babybear_rules.md`): `state_decode` /
`epilogue` word arithmetic as two 16-bit limbs (F4; only the word-in unit and the epilogue need it -- the state-chained VU
body is proven over 31-bit fields, so charging F4 to every unit is an overcount kept for safety), the signed-sum tightening,
and `TNORM_HI` (the `v_hi A(k)` product as a lookup; `checker_min.md` shows the product never exceeds p on admissible
inputs, so this too is conservative).  Stated, not verified as a gadget set: the lane brief's "take the modelled deltas and
state them".

## 3. Per batch (B = 4096, 393,216 units, 103 blocks)

Tables: everything the linear/quadratic tests touch is committed.  Per table row `v`: the multiplicity `m_T(v)` (1 `F_t`,
tableau 1) and the table-side inverse `h_v = m_T(v) / (z_T - t_v(beta_T))` (6 `F_t` coordinates, tableau 2) -- **7 committed
elements per table row, per proof**.  The table VALUES `t_v` are public coefficients of the table-side linear constraints
and are not committed (they enter `r^T A`, not `W`).

| table | rows | cols | queries/unit |
|---|---|---|---|
| SHIFT | 3,538,944 | 2 | 16 |
| TNORM | 1,048,576 | 2 | 2 |
| TNORM_HI (BabyBear repair) | 1,048,576 | 2 | 2 |
| SSHIFT_HI / SSHIFT_LO | 221,184 each | 2 | 2 + 2 |
| T_OP | 65,280 | 4 | 32 |
| T_range16 | 65,536 | 1 | 14 |
| LEAD | 32,768 | 3 | 4 |
| LEADNORM | 12,152 | 7 | 2 |
| ALIGN4 | 3,120 | 4 | 18 |
| T_HDR | 510 | 4 | 1 |
| **total** | **6,257,830** | | 95 |

~~~text
                                   rows        committed F_t          note
witness rows      331 x 103   =   34,093       130,134,528            tableau 1
helper rows       570 x 103   =   58,710       224,132,928            tableau 2, round 2 (after z_T, beta_T)
multiplicity rows 6.26M / 3840 =   1,630         6,257,830            tableau 1
table inverses  6 x 6.26M / 3840 = 9,778        37,546,980            tableau 2
blinding rows     3 per tableau =      6                              LDT, linear-test, quadratic-test rows (HVZK)
TOTAL                            104,217       398,092,426  = 388.8 kB / VU     (v1: 310,602 rows, 1,192.7M, 1,164.7 kB / VU)
                                                                                 tensor.py model at v2 (aligned triples): 105,312 rows, 394.9 kB / VU
~~~

**2.98x fewer rows than v1, 3.0x fewer committed bytes**; the fixed-table side is 11,408 rows = 11% of the matrix (v1: 121
rows).  Encoded codeword 6.83 GB (v1 20.4 GB).

### 3.1 Linear constraints and the powers-of-rho structure

| class | count | `r^T A` on the row |
|---|---|---|
| unit identities (11) + transition linkage (4), unit-major enumeration `u C + c` | 5,898,240 | `s_i G`, `G = (rho^(u C))_u` restricted to the block: **class 1** (all 34,093 witness rows; the neighbour terms fold in with `rho^(-C)`, as `LINTEST.md` 2.1) |
| per-table sum `sum_j h_j - sum_v h_v = 0` (6 coordinate equations x 11 tables) | 66 | constant `rho^(idx_T)` on every `h_j` coordinate slot: **class 2, the ONES row** (all 58,710 helper rows) |
| table-side `h_v (z_T - t_v(beta_T)) = m_T(v)` (6 `F_t` equations per table row) | 37,546,980 | on the multiplicity rows `-rho^(idx_v)`: geometric in `v` = class 1 again; on the 9,778 `h_v` rows the 6x6 multiplication-by-`(z_T - t_v(beta_T))` matrix times `rho^(idx_v)`: **varies with `v`: exceptional** |
| boundary (`row0_*`), last unit's missing successor | 2 | touch <= 2 rows per class: exceptional |
| **total** | **43,445,288** (v1: 335.5M) | **M = 2 fixed rows, X = 9,780 exceptional rows** (v1 chain layout: M = 3, X ~ 130) |

The lookup helper constraints do NOT add a class: with `y` and `z` virtual the helper identity is entirely a quadratic
triple, and the only linear constraint on `h_j` is the sum (a constant coefficient).  The exceptional rows are the
table-side inverses: 9,780 rows through the general path (14 transforms per row, 0.48 us/row measured: ~4.7 ms) -- the
one place the tables cost the linear test.  Soundness term: `(43.4M - 1) / |F_c| = 2^-160.6` (was `2^-163.7`).

### 3.2 Quadratic constraints

| class | rows in the test | per row per point | v1 |
|---|---|---|---|
| `F_t` triples `x o y = z`, 34 per unit | 34 x 103 = **3,502** rows (each 3 aligned rows) | 3 encoder calls + 6 mont-muls (`combine_quadratic`) | 33,485 product rows |
| `F_c` helper triples, 95 per unit | 95 x 103 = **9,785** rows (each 6 `h` coordinate rows + the query's `C` witness rows, `C` = the table's columns; 292 witness-row reads per unit-block) | `6 + C` encoder calls (or 0 with the cosets store) + `6C + 36 + 36 + 6` mont-muls (`combine_quadratic_fc`, `buckets_v2.py`) | inside the 33,485 (as `F_c x F_t`) |
| constraints | 393,216 x (34 + 95) = 50.7M | | 125.0M |

Soundness of the `F_c` product test: Ligero's quadratic test over the extension field (the code over `F_c` is the
coordinate-wise code over `F_t`, interpolation is linear); the `r_q` powers term is `(9,785 + 3,502 - 1) / |F_c| = 2^-172`.

### 3.3 Helpers to invert

`95 x 393,216 + 6,257,830 = 43,613,350` `F_c` elements (v1: 125,895,743).  **The inversion bucket shrinks 2.9x, not grows**: v1's
320 range-chunk lookups per unit were the helpers; v2's 89 (95) lookups into big tables replace them, and the 6.26M table
rows are far fewer than the 106M range chunks they replace.

### 3.4 HVZK

As v1: 256 random symbols per row (`104,217 x 256 x 4 B = 107 MB` of randomizers, v1 318 MB), 3 blinding rows per tableau,
HM96 leaf commitments per column (`zk_cost.json`: unchanged per column, the leaves are per column not per row).  All rows
above include the randomizer slots; nothing else is added for HVZK.  Malicious-verifier ZK (step 0) is unchanged.

## 4. Variant (b): public table coefficients (MODELLED, not the protocol as written)

The table side of the fractional identity is `sum_v m_T(v) / (z_T - t_v(beta_T))` with PUBLIC denominators, so the
`h_v` rows are unnecessary: the per-table sum becomes `sum_j h_j - sum_v c_v m_T(v) = 0` with public `c_v = 1 / (z_T -
t_v(beta_T))`.  Then 1 committed element per table row (94,439 rows, 352.1 kB / VU), 37.4M helpers to invert, X = 1,632
exceptional rows (the multiplicity rows, coefficient `rho^(idx) c_v`), 5.9M linear constraints -- but the VERIFIER computes
6.26M `F_c` inversions and the interpolants of 1,632 exceptional coefficient rows at the `t` opened columns per proof
(~1e10 `F_t` multiplications, seconds on a CPU core, milliseconds on a GPU).  Same soundness argument (Hab22 Lemma 5 on
the same rational identity).  Counted in `v2_shape.json["public_coefficients"]` and measured as the second row of each
bucket in `v2_bench.py`; not adopted here because it moves work to the verifier and changes `PROTOCOL.md` 4.2's row
classes -- a decision for the b-ligero2 lane and the red team.

## 5. Batch-size crossover (where v2 beats v1)

Rows: v1 = `75.83 B` (tables negligible); v2 = `22.66 B + 11,414` (witness + helper rows per VU, plus the fixed table side).
Row crossover `B* = 11,414 / (75.83 - 22.66) = 215 VUs`.  In measured time (`b_measured_4090_v2.md`): v2's fixed per-proof
cost is the table rows' commitment + the 6.26M table inversions + the exceptional linear-test rows; the per-VU marginal is
everything else / 4096; the crossover is where `fixed + marginal x B = v1's 28.4 us x B`.  Below it (a few hundred VUs) the
SHIFT table (57% of the table rows) should be replaced by its split-u variant (`checker_min.md` section 4: two 21k-row
lookups per product, +16 lookups per unit), which is a relation-preserving table change, not a new predicate.

## 6. Conventions for the b-ligero2 port (in addition to `README.md`'s b-commit / b-lintest lists)

* **Transposed layout**: row `(AIR column j, block b)`; block = 3840 consecutive units; unit `u` at slot `u - 3840 b`.
  Tableau 1: the 331 witness columns then the multiplicity rows; tableau 2: the 570 helper coordinate rows (`h_{j,c}`,
  coordinate-major: the 6 coordinate rows of lookup `j` are consecutive) then the 9,778 table-inverse rows.
* **Lookup query encoding**: `w_j(beta_T) = sum_{c < C} beta_T^c w_{j,c}` over the table's columns in `fixed_tables()` order
  (key first); `beta_T, z_T in F_c` per table (one message, redraw slots as `PROTOCOL.md` step 2).
* **Helper identity as an `F_c` triple with virtual `y`, `z`**: the prover's quadratic test runs `combine_quadratic_fc`
  per table group (`C` compiled in); the verifier's column check evaluates `y = z_T ONE(eta) - sum_c beta^c W_{j,c}(eta)`
  from the opened witness rows and `z = ONE(eta)` (`ONE` = interpolant of `[j < l]`).
* **Constraint enumeration**: unit-major blocks of `C = 15` (11 identities + 4 linkage) then the 11 per-table sums, then
  the table-side equations in table order, row order; `r_add = rho^idx`.  `M = 2` (`G`, `ONES`), exceptional = the table
  inverse rows + the two boundary rows.
* **Inversions**: one batch per table over `queries + |T|` elements; SHIFT alone is 6.3M + 3.5M.
* **Tables are per proof**: the 6.26M rows' multiplicities are witness-dependent (round 1), the inverses challenge-dependent
  (round 2); nothing about them can be preprocessed under the "no trusted setup / no preprocessing" policy beyond the
  public `t_v(beta_T)` evaluation the verifier does anyway.
