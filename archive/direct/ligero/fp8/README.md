# fp8-ada: B-Ligero for the Ada FP8 tensor-core step (`sm89.mma.m16n8k32.e4m3`)

**Relation.** `GemmCoordinate<K=1536>`: 48 chained transitions `c_{s+1} = tc_dot(ADA_E4M3_M16N8K32, c_s, a[32s:32s+32], b[32s:32s+32])`
from `c_0 = +0`; the final FP32 accumulator word is the public claim (packed to 22 bits: sign, 8-bit exponent field, the 13
kept fraction bits -- the low 10 bits of every honest word are zero because the adder is 14 bits wide; a claim with nonzero
low bits is rejected by the verifier's decode).  Per step: 32 exact E4M3 products (4x4-bit significands, exponents
`max(t,1) - 7` each, subnormals unnormalised), two groups of 16 on a 14-bit truncating adder with exponent floor -139,
the incoming accumulator joining the first group and the group result the second; normalisation / subnormal shift /
overflow exactly as `verity.ml.circuits.silicon.silicon.group_sum`.  Domain: finite operands (`0x7f`/`0xff` NaN rejected) and finite
accumulators; a group leaving the FP32 range gives the sticky infinity as `tc_dot` does.

**Gadget layout** (`relation.py`, on `compile.LigeroCtx`; the BF16 gadgets of `verity_numerical.checker.gadgets` reused
unchanged on `FP8Params`, an adder-side view with `sig = 4`, `acc_sig = width = 14`, `exp = 8` (FP32 e_min/e_max)):

* operands: 3 pinned rows each (`s, m, e`; the verifier decodes the public bytes; no bit rows);
* products: `Product` (zero flag, 8-bit range of `u`), 32 per step;
* per group: `GroupMax` (18 selectors, 9-bit shifts), `AlignState` (POW table, 14-bit q/r), 16 x `AlignProduct` (ALIGN
  table on `u 2^7`, 8-bit q/r), `SignedSum` (20-bit `v`), `LeadingBit`, `Normalise` (`M14 = floor(v 2^(E - e'))`, the
  driver's `vbound` theorem check), `StateOut` (zero / overflow flags);
* **the lossy rescale (new, zero extra gadget classes):** the accumulator's fraction `f` is range-checked by 23 bit rows
  anyway; the adder's 14-bit state is the affine expression `Mt = sum_{i>=10} 2^(i-10) f_i + nz 2^13` of those rows (the
  ten dropped bits are the ten low rows).  The zero flag stays on the full 24-bit `M` (`z M = 0`, `(M + z) inv = 1`), and
  the exponent still enters the group maximum -- as silicon does.  Unit mode pins `Mt` directly;
* pack: `hidden 2^13 + low = M14`, `f = low 2^10`, components `y.s, y.t, y.f`; chain mode links `(s, t, f)` across columns
  (`chain.py`, unchanged) and tests the packed word at chain ends.

Census (chain unit): 3,769 rows -- 161 hint, 2,474 bit, 640 selector, 301 product, 1 inverse, 192 pin; 589 linear,
3,532 quadratic (BF16 unit: 3,516 rows).  `python -m archive.direct.ligero.fp8.census` prints it next to BF16.

**Protocol hooks.** `protocol.RelationHooks` (public decode, statement word widths, domain tag) with the BF16 default
byte-identical to the pre-hook transcript; `run.py --relation fp8-ada` dispatches `gate-vu` / `bench-vu` here.

**Run.**

~~~
uv run pytest archive/direct/ligero/fp8 -q                      # 2,000 honest cases, 52 wrong claims, row mutations, CPU chain (non-ZK + ZK)
python -m archive.direct.ligero.run --relation fp8-ada gate-vu  --device cuda --vus 64
python -m archive.direct.ligero.run --relation fp8-ada bench-vu --device cuda --batch 4096 --total-vus 4096 --reps 3 [--zk --mode fiat-shamir] --out result.json
research run --on vy-sp1 --project verity --campaign r21-silicon --tool bench_vu_fp8 --scratch triton --source . -- python -m archive.direct.ligero.run --relation fp8-ada bench-vu ...
~~~
