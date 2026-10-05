import Mathlib.Tactic.FieldSimp
import FlockSoundness.Protocol.ZK.Table

/-!
# Zero knowledge of one masked table: Lemmas A and B, and `table_shvzk`

For the model of `Table` (the reference prover `Table.view` and `S_shvzk` as `Table.sim`), at a fixed non-degenerate coin
vector:

* **The identity (★)** (§4.3), `star`: rep `r`'s two extra-lane entries at an opened position are determined by the
  witness lanes' entries there, `y₁'^r` and `ȳ^r`, because `W^r` is invertible.
* **Lemma B's rows** (§4.3), `card_realCore` and `card_simCore`: the prover's view is a public function (`psi`) of four
  components, each of `card_seq_masked`'s form: anything computed from earlier blocks plus a surjective image of a fresh
  block. Row (i) is the masked `ŝ` (the mask slot, onto by the rank check); rows (ii) and (iii) the witness lanes' entries
  at the opened positions with `(y₁'^r, ȳ^r)` (the paddings, onto by T4, and the extra lanes, onto because `W^r` is);
  rows (vi) and (vii) the masked values with `ρ_in`, `σ_in` (the pads); rows (viii)–(x) `C_h`, `Y₂`, `Y₁` (`pad_h` onto by
  T4, `pad_μ` and `μ` scaled by `β ≠ 0`), with the salts and ideal-leaf seeds. Rows (iv), (v), (xi) and (xii) are
  `psi`'s: the extra-lane entries by (★), levels ≥ 1 from `y₁'^r` and fresh salts, `τ` and `C_μ` from the replayed
  constraint, the leaves.
* **`table_shvzk`** (Lemma B in world `W₁`): the real view and `S_shvzk(x, e)` have the same distribution.
* **Lemma A at a non-degenerate coin vector** (§4.2), `table_prefinal_translate` and `table_prefinal_indep`: in world
  `W₀`, the pre-final views on two witnesses carrying the public region data agree along a translation of the blocks `u`,
  `R`, `h`, `μ` that reads only earlier blocks, so by `card_fiber_eq_of_triShift` they have the same distribution.

Everything here is about one table, at coins where the prover refuses nothing. The paper's lemmas are per session of `J`
tables: the product over tables (no claim or message reads two tables' witnesses or masks) is `Session`'s; Lemma A's
cases where the coins make the prover refuse (a failing rank check, dependent `ρ^r`, `β = 0`), with the schedule, are
`Refusal`'s; and Lemma C, which T5 builds from them, is `AdaptivePrefinal`'s.
-/

namespace FlockSoundness.ZK

open Finset

-- The randomness and the view's components are products of up to fifteen factors, and their `Fintype`, `DecidableEq`
-- and group instances exceed instance search's default size.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL)

/-! ## The identity (★) -/

/-! ## The view as a function of four masked components -/

/-! ## The masks -/

/-- Rows (ii) and (iii)'s mask: the paddings at the opened positions, `W^r` of the extra lanes, and `ȳ^r`. -/
def mask1 : ((Fin T.nL → Fin T.tpad → F) × T.XLanes × T.XPad) →+
    ((Fin T.nS → Fin T.nL → F) × (Fin 2 → Fin T.L → K) × (Fin 2 → Fin T.tpad → K)) where
  toFun b := (fun q l => ∑ j, b.1 l j * T.Gp q j, fun r c => T.Wr r (b.2.1 r 0 c) (b.2.1 r 1 c),
    fun r j => ∑ l, T.rbar r l * T.embed (b.1 l j) + T.Wr r (b.2.2 r 0 j) (b.2.2 r 1 j))
  map_zero' := by
    simp only [Prod.fst_zero, Prod.snd_zero, Pi.zero_apply, zero_mul, Finset.sum_const_zero, map_zero, mul_zero,
      Wr, add_zero]
    rfl
  map_add' b b' := by
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · funext q l
      simp only [Prod.fst_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]
    · funext r c
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add]
    · funext r j
      simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, Wr_add, map_add, mul_add, Finset.sum_add_distrib]
      ring

variable [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-! ## Lemma B: the real view and `S_shvzk` -/

variable [CharP F 2]

/-! ## Lemma A: the pre-final view doesn't depend on the witness -/

/-! ## Counting -/

variable [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed]
  [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]

instance : Fintype T.Seeds := inferInstance
instance : Fintype T.Rand := inferInstance
instance : Fintype T.SimRand := inferInstance

end Table

end FlockSoundness.ZK
