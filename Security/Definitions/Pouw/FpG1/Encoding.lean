import Mathlib

/-!
# FP8-g1 and W4A8-g1: the operand encodings (trusted)

FPX-DESIGN's g1 designs (note:20261019T0817Z-report-fpx-design; `benchmarks/ppl/steps.py`'s `native_linear` with `act`
or `weight` = `g1`, PR #2710). Each GEMM operand is an integer level `R` that the honest prover picks on a format's
integer grid, plus a live fair coin `b`, served centred as `R + b − ½`; the int8 MMA multiplies the integers `R + b`,
and the epilogue takes the coin's two rank-one terms off (`Proofs.Pouw.FpG1.Gemm`).

* `E4M3Nat`, `E2M1x2Nat`: E4M3's non-negative integers (at most four significant bits, at most 448, its largest finite
  value) and twice E2M1's non-negative values (at most two significant bits, at most 12: `{0, 1, 2, 3, 4, 6, 8, 12}`).
* `grid`, `e4m3Grid B`, `e2m1x2Grid`: a format's integers in `[−B, B]`, as `steps.py`'s `LEVELS` (`e4m3i120`,
  `e4m3i60`, `e2m1x2`).
* `rn`: round to nearest onto a grid, ties to the lower level (`steps.py`'s `to_grid` in mode `round`: the level
  below when `v − lo ≤ hi − v`). It is the honest encoder's choice only. No check reads it: GPTQ and a clip search
  pick other levels on the same grid, and the Program checks the level's grid, not how it was picked.
* `Coin`: a live fair coin, `0` or `1`, the record's operand bit (co32c's).
* `Entry`: an operand entry as a proof unit's values carry it: FP8-g1's E4M3 level and coin, W4A8-g1's weight (an
  integer UE4M3 block scale `m`, a doubled E2M1 level `q` and a coin), or the record's Dither int8 activation (`⌊v⌋`
  and its coin, served uncentred). `Entry.int` is what the int8 MMA reads, `Entry.offset` the centring (`½`, `½`,
  `0`) and `Entry.served = Entry.int − Entry.offset` the value the served model multiplies.
* `Grids`, `Grids.On`: a device's g1 grids (sm_120: activations to 120, weights to 60, block scales to 5; B200 and
  H200 on Dither-2S: 120, 120 and 10) and an entry on them at its side: a level of its format's grid plus a coin.
  sm_120's 60 and 5 are FPX-DESIGN's, derived for Dither-C's co32c (7-bit weights), not for sm_120's record H1
  (co32c-u16, weights to ±110), so on H1 they are conservative; g1's values fit both (RT2-FP-2,
  note:20261019T1100Z-draft-rt2-fp). No Program checks `Grids.On`: it describes what an honest g1 prover commits.
-/

namespace Pouw.FpG1

/-- **E4M3's non-negative integers**: `n ≤ 448` (E4M3's largest finite value) with at most four significant bits,
`n = m·2^e` for some `m < 16`: every integer to 16, then E4M3's spacing (2 to 32, 4 to 64, 8 to 128, 16 to 256, 32 to
448), as `steps.py`'s `_e4m3_integers`. -/
def E4M3Nat (n : ℕ) : Prop := n ≤ 448 ∧ ∃ e ∈ Finset.range 9, 2 ^ e ∣ n ∧ n / 2 ^ e < 16

instance instDecidablePredE4M3Nat : DecidablePred E4M3Nat := fun n =>
  inferInstanceAs (Decidable (n ≤ 448 ∧ ∃ e ∈ Finset.range 9, 2 ^ e ∣ n ∧ n / 2 ^ e < 16))

/-- **Twice E2M1's non-negative values**: `n ≤ 12` with at most two significant bits, `{0, 1, 2, 3, 4, 6, 8, 12}`
(E2M1 is `{0, ½, 1, 1½, 2, 3, 4, 6}`). -/
def E2M1x2Nat (n : ℕ) : Prop := n ≤ 12 ∧ ∃ e ∈ Finset.range 4, 2 ^ e ∣ n ∧ n / 2 ^ e < 4

instance instDecidablePredE2M1x2Nat : DecidablePred E2M1x2Nat := fun n =>
  inferInstanceAs (Decidable (n ≤ 12 ∧ ∃ e ∈ Finset.range 4, 2 ^ e ∣ n ∧ n / 2 ^ e < 4))

/-- **A format's integer grid to `B`**: the integers in `[−B, B]` whose magnitude is one of the format's (`P`). -/
def grid (P : ℕ → Prop) [DecidablePred P] (B : ℕ) : Finset ℤ :=
  (Finset.Icc (-(B : ℤ)) B).filter fun z => P z.natAbs

/-- **E4M3's integers to `B`** (`steps.py`'s `e4m3i120` at `B = 120`, `e4m3i60` at `B = 60`). -/
def e4m3Grid (B : ℕ) : Finset ℤ := grid E4M3Nat B

/-- **The doubled E2M1 grid** (`steps.py`'s `e2m1x2`): `{0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}`. -/
def e2m1x2Grid : Finset ℤ := grid E2M1x2Nat 12

/-- A live fair coin: `0` or `1`. -/
abbrev Coin : Type := Fin 2

/-- **An operand entry** as a proof unit's values carry it:
* `fp8 R b`: FP8-g1, an E4M3 integer level `R` and its coin, served `R + b − ½`;
* `fp4 m q b`: W4A8-g1's weight, an integer UE4M3 block scale `m`, a doubled E2M1 level `q` and its coin, served
  `m·q + b − ½`;
* `int8 R b`: the record's Dither activation beside W4A8-g1's weights, `R = ⌊v⌋` and its coin, served `R + b`. -/
inductive Entry
  | fp8 (level : ℤ) (coin : Coin)
  | fp4 (scale : ℕ) (level : ℤ) (coin : Coin)
  | int8 (level : ℤ) (coin : Coin)

/-- A GEMM operand's side: activations (the left operand) or weights (the right). -/
inductive Side
  | act
  | wt

/-- **A device's g1 grids**: E4M3's bound for activations and for weights, and W4A8-g1's largest block scale. -/
structure Grids where
  act : ℕ
  wt : ℕ
  scale : ℕ

end Pouw.FpG1
