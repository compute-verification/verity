# BabyBear embedding rules for the WP2 checker (red-team-babybear)

p = 2^31 - 2^27 + 1 = 2013265921 (log2 p = 30.907). Every predicate of every gadget contract was run on
integer intervals (`verity_numerical.redteam.babybear.analyse`): inputs and hints at their declared ranges, table
columns at their row extremes; the mod-p reading of a predicate is sound iff the interval of what it constrains lies
in a window of width p (identity: `a - b` in (-p, p); range check on an expression: in `[hi - p, lo + p)`; the
product realisation of a variable bound: `x * ipow` in `[2^B - p, p)`). Widths: acc_bits = 25, prod_bits = 26, s_bits = 30, |S| <= s_max = 570417150 (2^29.x), norm remainder 7 bits, k in [-7, 24].

## 1. Predicates whose integer bound exceeds p as stated, and the rule that closes each

| gadget | predicate | as-stated interval | bits | rule | embedded bits | embedded sound | added columns / lookups |
|---|---|---|---|---|---|---|---|
| fp32_decode | `acc.word` | [-4294967295, 4286578687] | 32 | LIMB16 | 16 | yes | +2 hint columns (f_lo 16 b, f_hi 7 b), +2 16/7-bit ranges, -1 23-bit range, +1 identity; c as 2 public limbs |
| align_state | `align_state.rem` | [0, 1125899873288192] | 50 | LT_DIRECT | 25 | yes | no multiplication; one 25-bit range (2 chunk lookups) on a linear wire, as before |
| align_state | `align_state.quot` | [0, 1125899873288192] | 50 | LT_DIRECT+QUOT_TIGHT | 25 | yes | no multiplication; one 25-bit range on a linear wire |
| align_product | `align_prod.rem` | [0, 4294901760] | 32 | LT_DIRECT | 16 | yes | no multiplication; one 16-bit range on a linear wire |
| align_product | `align_prod.quot` | [0, 4294901760] | 32 | LT_DIRECT+QUOT_TIGHT | 16 | yes | no multiplication; one 16-bit range on a linear wire |
| leading_bit | `lead.window` | [-576460752303423488, 1152921503533105152] | 60 | LT_DIRECT+V_PREMISE | 30 | yes | LEAD gains column span; no multiplication; two 30-bit ranges (4 chunk lookups) on the linear wire v - lo (>= 0 and < span) |
| normalise | `norm.div` | [-18014398492704768, 2147483647] | 54 | NORM_MBND+NORM_VBND(redundant)+V_PREMISE | 31 | yes | NORM gains columns mbnd (and vbnd); +1 30-bit range (2 chunk lookups) per group [+1 if vbnd kept]; 0 hints, 0 multiplications |
| pack | `pack.y[0]` | [-5351931903, 5435817983] | 33 | LIMB16 | 16 | yes | +2 hint columns (low_lo, low_hi), +1 identity; y as 2 public limbs |
| epilogue | `cast.split` | [-4294967295, 4294967295] | 32 | LIMB16 | 0 | yes | -2 hint columns (hi, lo are the public limbs), -1 identity, -2 ranges |

## 2. The rules

- **LIMB16**: the word is two public 16-bit limbs; the identity is stated per limb (fraction limbs f_lo/f_hi or low_lo/low_hi as hints)
- **LT_DIRECT**: direct variable-bound check rng(bnd - 1 - x, 0, 2^B) against the table's bound column (POW/ALIGN: the partner column; LEAD: new column span), plus rng(x, 0, 2^B) when x is not a range-checked hint (the leading-bit window v - lo: the direct form alone loses v >= lo -- z3 found it)
- **QUOT_TIGHT**: the tight quotient bound q < 2^B / pow (direct) keeps q * pow + r below 2^B, so the division identity stays inside (-p, p)
- **NORM_MBND**: NORM column mbnd = 2^30 / div and rng(mbnd - 1 - M', 0, 2^30): the quotient is tight relative to the dividend v < 2^30
- **NORM_VBND**: NORM column vbnd = 2^24 / mul (k >= 0) and rng(vbnd - 1 - v, 0, 2^30): v * mul < 2^24 (redundant given leading_bit; see z3)
- **V_PREMISE**: v is the signed-sum wire, |v| <= s_max = 570,417,150 < 2^29.1 by the ranges of its terms; leading_bit/normalise are stated for v in [0, s_max]

## 3. Safe as stated over BabyBear (bound < p): no change for the lanes

| gadget | predicate | kind | interval | bits |
|---|---|---|---|---|
| bf16_decode | `s` | boolean | [0, 1] | 1 |
| bf16_decode | `nz` | boolean | [0, 1] | 1 |
| bf16_decode | `bf16.t` | rng | [0, 254] | 8 |
| bf16_decode | `bf16.f` | rng | [0, 127] | 7 |
| bf16_decode | `bf16.word` | eq | [-65535, 65407] | 16 |
| bf16_decode | `bf16.nz0` | eq | [0, 254] | 8 |
| bf16_decode | `bf16.nz1` | rng | [-1, 254] | 8 |
| fp32_decode | `s` | boolean | [0, 1] | 1 |
| fp32_decode | `nz` | boolean | [0, 1] | 1 |
| fp32_decode | `z` | boolean | [0, 1] | 1 |
| fp32_decode | `acc.t` | rng | [0, 254] | 8 |
| fp32_decode | `acc.f` | rng | [0, 8388607] | 23 |
| fp32_decode | `acc.nz0` | eq | [0, 254] | 8 |
| fp32_decode | `acc.nz1` | rng | [-1, 254] | 8 |
| fp32_decode | `acc.z0` | eq | [0, 16777215] | 24 |
| fp32_decode | `acc.z1` | rng | [-1, 16777215] | 24 |
| product | `s_a` | boolean | [0, 1] | 1 |
| product | `s_b` | boolean | [0, 1] | 1 |
| product | `z` | boolean | [0, 1] | 1 |
| product | `prod.z0` | eq | [0, 65025] | 16 |
| product | `prod.z1` | rng | [-1, 65025] | 16 |
| group_max | `sel[0]` | boolean | [0, 1] | 1 |
| group_max | `sel[1]` | boolean | [0, 1] | 1 |
| group_max | `sel[2]` | boolean | [0, 1] | 1 |
| group_max | `sel[3]` | boolean | [0, 1] | 1 |
| group_max | `sel[4]` | boolean | [0, 1] | 1 |
| group_max | `sel[5]` | boolean | [0, 1] | 1 |
| group_max | `sel[6]` | boolean | [0, 1] | 1 |
| group_max | `sel[7]` | boolean | [0, 1] | 1 |
| group_max | `sel[8]` | boolean | [0, 1] | 1 |
| group_max | `sel[9]` | boolean | [0, 1] | 1 |
| group_max | `max.E` | rng | [-132, 259] | 9 |
| group_max | `max.attained` | eq | [-2659, 2463] | 12 |
| group_max | `max.d[0]` | rng | [-391, 511] | 9 |
| group_max | `max.d[1]` | rng | [-391, 511] | 9 |
| group_max | `max.d[2]` | rng | [-391, 511] | 9 |
| group_max | `max.d[3]` | rng | [-391, 511] | 9 |
| group_max | `max.d[4]` | rng | [-391, 511] | 9 |
| group_max | `max.d[5]` | rng | [-391, 511] | 9 |
| group_max | `max.d[6]` | rng | [-391, 511] | 9 |
| group_max | `max.d[7]` | rng | [-391, 511] | 9 |
| group_max | `max.d[8]` | rng | [-391, 511] | 9 |
| group_max | `max.d_floor` | rng | [0, 391] | 9 |
| align_state | `align_state.q` | rng | [0, 33554431] | 25 |
| align_state | `align_state.r` | rng | [0, 33554431] | 25 |
| align_state | `align_state.pow` | lookup | [0, 511] | 9 |
| align_state | `align_state.div` | eq | [-33554430, 67108862] | 26 |
| align_product | `align_prod.q` | rng | [0, 65535] | 16 |
| align_product | `align_prod.r` | rng | [0, 65535] | 16 |
| align_product | `align_prod.align` | lookup | [0, 511] | 9 |
| align_product | `align_prod.div` | eq | [-65535, 131070] | 17 |
| signed_sum | `s[0]` | boolean | [0, 1] | 1 |
| signed_sum | `s[1]` | boolean | [0, 1] | 1 |
| signed_sum | `s[2]` | boolean | [0, 1] | 1 |
| signed_sum | `s[3]` | boolean | [0, 1] | 1 |
| signed_sum | `s[4]` | boolean | [0, 1] | 1 |
| signed_sum | `s[5]` | boolean | [0, 1] | 1 |
| signed_sum | `s[6]` | boolean | [0, 1] | 1 |
| signed_sum | `s[7]` | boolean | [0, 1] | 1 |
| signed_sum | `s[8]` | boolean | [0, 1] | 1 |
| signed_sum | `sgn` | boolean | [0, 1] | 1 |
| signed_sum | `z` | boolean | [0, 1] | 1 |
| signed_sum | `sum.v` | rng | [-570425335, 570425335] | 30 |
| signed_sum | `sum.z0` | eq | [-570425335, 570425335] | 30 |
| signed_sum | `sum.z1` | rng | [-570425336, 570425335] | 30 |
| leading_bit | `lead.lead` | lookup | [0, 30] | 5 |
| normalise | `clamp` | boolean | [0, 1] | 1 |
| normalise | `norm.M` | rng | [0, 16777215] | 24 |
| normalise | `norm.r` | rng | [0, 127] | 7 |
| normalise | `norm.clamp_proof` | rng | [-391, 390] | 9 |
| normalise | `norm.norm` | lookup | [-7, 384] | 9 |
| normalise | `norm.rem` | lt_pow | [0, 16256] | 14 |
| state_out | `sgn` | boolean | [0, 1] | 1 |
| state_out | `z` | boolean | [0, 1] | 1 |
| state_out | `ovf` | boolean | [0, 1] | 1 |
| state_out | `out.z0` | eq | [0, 16777215] | 24 |
| state_out | `out.z1` | rng | [-1, 16777215] | 24 |
| state_out | `out.ovf_proof` | rng | [-254, 253] | 8 |
| pack | `s` | boolean | [0, 1] | 1 |
| pack | `z` | boolean | [0, 1] | 1 |
| pack | `ovf[0]` | boolean | [0, 1] | 1 |
| pack | `ovf[1]` | boolean | [0, 1] | 1 |
| pack | `sgn[0]` | boolean | [0, 1] | 1 |
| pack | `sgn[1]` | boolean | [0, 1] | 1 |
| pack | `hidden` | boolean | [0, 1] | 1 |
| pack | `pack.low` | rng | [0, 8388607] | 23 |
| pack | `pack.M` | eq | [-16777215, 16777215] | 24 |
| epilogue | `lsb` | boolean | [0, 1] | 1 |
| epilogue | `carry` | boolean | [0, 1] | 1 |
| epilogue | `cast.hi` | rng | [0, 65535] | 16 |
| epilogue | `cast.lo` | rng | [0, 65535] | 16 |
| epilogue | `cast.hi2` | rng | [0, 32767] | 15 |
| epilogue | `cast.rem` | rng | [0, 65535] | 16 |
| epilogue | `cast.lsb_split` | eq | [-65535, 65535] | 16 |
| epilogue | `cast.round` | eq | [-98303, 98304] | 17 |
| epilogue | `cast.y16` | rng | [0, 65536] | 17 |

## 4. Under the full rule set every one of the 109 predicates lies inside its window.

## 5. Dropping one rule at a time: predicates that leave their window (interval verdict; z3 in `z3_babybear` decides)

| rule dropped | predicates outside the window by interval arithmetic |
|---|---|
| LIMB16 | `fp32_decode.acc.word`, `pack.pack.y[0]`, `epilogue.cast.split` |
| LT_DIRECT | `align_state.align_state.rem`, `align_state.align_state.quot`, `align_product.align_prod.rem`, `align_product.align_prod.quot`, `leading_bit.lead.window` |
| QUOT_TIGHT | `align_state.align_state.div`, `align_product.align_prod.div` |
| NORM_MBND | `normalise.norm.div` |
| NORM_VBND | `normalise.norm.div` |
| V_PREMISE | `leading_bit.lead.window`, `normalise.norm.vbnd` |

