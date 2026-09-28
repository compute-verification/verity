import Proofs.Flock.Soundness.ZK.SHVZK
import Proofs.Flock.Soundness.ZK.Hiding
import Proofs.Flock.Soundness.ZK.RealView

/-!
# Zero knowledge: Lemma B with real hm96 leaves

`table_shvzk` is exact in world `W₁`, where every leaf the final message doesn't open is ideal. The real prover's leaves
are hm96 leaves of what they commit: the level-0 codeword's columns at the other positions, and each pads tree's columns
at the positions its rep doesn't open. `S_shvzk`'s are hm96 leaves too: its dummy run's level-0 columns, and zero pads
columns (§3.2 step 6). This file moves both sides to `W₁` with `ideal_leaves_swap`, at `δ₁` per hidden leaf, which is
Lemma B's bound (§4.3), for one table.

The bound is `2·|Hid|·δ₁`, with `Table.Hid` the leaves the final message doesn't open. The paper's `N_hid` counts
every level-0 and pads-tree leaf, so `|Hid| ≤ N_hid`, and the Lean bound is at least as strong as the paper's
`2·δ₁·N_hid`.

* `Table.Hm96`, `Table.viewWith`, `Table.simWith`, `Hm96.viewR`, `Hm96.simR` (in `RealView`): hm96's structure, the views
  with their hidden leaves given, and the two views with real leaves; `view` and `sim` are `viewWith`, `simWith` at ideal
  leaves.
* `table_shvzk_hm96`: for every event on the view, the real prover's probability and `S_shvzk`'s differ by at most
  `2·|Hid|·δ₁` in each direction.
-/

namespace FlockSoundness.ZK

open Finset Game
open scoped ENNReal

-- As in `SHVZK`: the randomness is a product of up to fifteen factors.
set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **T1 over any finite set of leaves**: `ideal_leaves_swap` through an enumeration of the index. -/
theorem ideal_leaves_swap_fintype {Y B C D ι : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
    [Fintype ι] [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁)
    (H : B × C → D) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → ι → B) (G : Ω → (ι → D) → Prop) :
    prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
      ≤ prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + Fintype.card ι * δ₁ := by
  let e := Fintype.equivFin ι
  have h := ideal_leaves_swap My c hT1 H (Fintype.card ι) (Ω := Ω) (fun ω j => x ω (e.symm j))
    (fun ω L => G ω fun i => L (e i))
  have hl : prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
      = prCoin (fun p : Ω × (Fin (Fintype.card ι) → Y) =>
          G p.1 fun i => H (x p.1 (e.symm (e i)) + My (p.2 (e i)), c (p.2 (e i)))) := by
    refine Eq.trans ?_ (prCoin_congr_equiv ((Equiv.refl Ω).prodCongr (e.arrowCongr (Equiv.refl Y))) _)
    congr 1
    funext p
    simp [Equiv.arrowCongr_apply]
  have hr : prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
      = prCoin (fun p : Ω × (Fin (Fintype.card ι) → B × Y) => G p.1 fun i => H ((p.2 (e i)).1, c (p.2 (e i)).2)) := by
    refine Eq.trans ?_ (prCoin_congr_equiv ((Equiv.refl Ω).prodCongr (e.arrowCongr (Equiv.refl (B × Y)))) _)
    congr 1
    funext p
    simp [Equiv.arrowCongr_apply]
  rw [hl, hr]
  exact h

/-- Equal distributions give equal probabilities to every event. -/
theorem SameDist.prCoin_eq {Ω Ω' V : Type} [Fintype Ω] [Fintype Ω'] [Nonempty Ω] [Nonempty Ω'] {f : Ω → V}
    {g : Ω' → V} (h : SameDist f g) (E : V → Prop) : prCoin (fun ω => E (f ω)) = prCoin (fun ω' => E (g ω')) := by
  classical
  have hc := h.comp (fun v => decide (E v)) true
  simp only [decide_eq_true_eq] at hc
  rw [prCoin_eq_card, prCoin_eq_card]
  have h1 : (Fintype.card Ω : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have h2 : (Fintype.card Ω' : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  rw [ENNReal.div_eq_div_iff h2 (ENNReal.natCast_ne_top _) h1 (ENNReal.natCast_ne_top _)]
  have e : #{ω | E (f ω)} * Fintype.card Ω' = #{ω' | E (g ω')} * Fintype.card Ω := by convert hc using 2
  rw [mul_comm, ← Nat.cast_mul, e, Nat.cast_mul, mul_comm]

/-- An event of the first coordinate of a product of uniform coins has its probability over that coordinate. -/
theorem prCoin_fst {C E : Type} [Fintype C] [Fintype E] [Nonempty E] (P : C → Prop) :
    prCoin (fun p : C × E => P p.1) = prCoin P := by
  rw [prCoin_prod_split (fun c (_ : E) => P c)]
  unfold prCoin
  congr 1
  funext c
  exact avg_const _

/-- Complements: `Pr[¬P] = 1 − Pr[P]`, as a sum. -/
theorem prCoin_add_not {C : Type} [Fintype C] [Nonempty C] (P : C → Prop) : prCoin P + prCoin (fun c => ¬ P c) = 1 := by
  classical
  unfold prCoin
  rw [← avg_add]
  calc avg _ = avg (fun _ : C => (1 : ℝ≥0∞)) := by
        congr 1
        funext c
        by_cases hc : P c <;> simp [hc]
    _ = 1 := avg_const _

theorem prCoin_ne_top {C : Type} [Fintype C] [Nonempty C] (P : C → Prop) : prCoin P ≠ ∞ :=
  ne_top_of_le_ne_top ENNReal.one_ne_top (le_of_le_of_eq le_self_add (prCoin_add_not P))

/-- **T1 the other way**: ideal leaves are within `|ι|·δ₁` of real ones, by `ideal_leaves_swap_fintype` on the
complement. -/
theorem real_leaves_swap_fintype {Y B C D ι : Type} [Fintype Y] [Nonempty Y] [AddCommGroup B] [Fintype B]
    [Fintype ι] [DecidableEq ι] (My : Y → B) (c : Y → C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding My c δ₁)
    (H : B × C → D) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → ι → B) (G : Ω → (ι → D) → Prop) :
    prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
      ≤ prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i))) + Fintype.card ι * δ₁ := by
  have : Nonempty B := ⟨0⟩
  have h := ideal_leaves_swap_fintype My c hT1 H x (fun ω L => ¬ G ω L)
  have a1 := prCoin_add_not (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
  have a2 := prCoin_add_not (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
  have hfin := prCoin_ne_top (fun p : Ω × (ι → B × Y) => ¬ G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
  refine (ENNReal.add_le_add_iff_right hfin).1 ?_
  calc prCoin (fun p : Ω × (ι → B × Y) => G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
        + prCoin (fun p : Ω × (ι → B × Y) => ¬ G p.1 fun i => H ((p.2 i).1, c (p.2 i).2))
      = prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        + prCoin (fun p : Ω × (ι → Y) => ¬ G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i))) := by rw [a1, a2]
    _ ≤ prCoin (fun p : Ω × (ι → Y) => G p.1 fun i => H (x p.1 i + My (p.2 i), c (p.2 i)))
        + (prCoin (fun p : Ω × (ι → B × Y) => ¬ G p.1 fun i => H ((p.2 i).1, c (p.2 i).2)) + Fintype.card ι * δ₁) :=
        add_le_add le_rfl h
    _ = _ := by ring

namespace Table

variable {F K W Salt B D Hi VL : Type} [Field F] [Field K] [AddCommGroup B]
  (T : Table F K W Salt (B × Salt) D Hi VL)

/-- The hidden leaves' seeds in the prover's randomness. -/
def seedsOf (s : T.Seeds) : T.Hid → B × Salt := Sum.elim s.2.2.1 fun rk => s.2.2.2.2.2.1 rk.1 rk.2

/-- Replace the hidden leaves' seeds. -/
def setSeeds (s : T.Seeds) (u : T.Hid → B × Salt) : T.Seeds :=
  (s.1, s.2.1, fun q => u (.inl q), s.2.2.2.1, s.2.2.2.2.1, fun r k => u (.inr (r, k)), s.2.2.2.2.2.2)

omit [AddCommGroup B] in
theorem seedsOf_setSeeds (s : T.Seeds) (u : T.Hid → B × Salt) : T.seedsOf (T.setSeeds s u) = u := by
  funext i
  rcases i with q | ⟨r, k⟩ <;> rfl

omit [AddCommGroup B] in
theorem setSeeds_seedsOf (s : T.Seeds) : T.setSeeds s (T.seedsOf s) = s := rfl

omit [AddCommGroup B] in
theorem setSeeds_setSeeds (s : T.Seeds) (u v : T.Hid → B × Salt) : T.setSeeds (T.setSeeds s u) v = T.setSeeds s v := rfl

/-- The prover's randomness with the hidden leaves' seeds replaced. -/
def withSeeds (ω : T.Rand) (u : T.Hid → B × Salt) : T.Rand :=
  (ω.1, ω.2.1, ω.2.2.1, ω.2.2.2.1, ω.2.2.2.2.1, ω.2.2.2.2.2.1, T.setSeeds ω.2.2.2.2.2.2 u)

/-- `S_shvzk`'s randomness with its dummy run's hidden leaves' seeds replaced. -/
def simWithSeeds (σ : T.SimRand) (u : T.Hid → B × Salt) : T.SimRand :=
  (σ.1, σ.2.1, σ.2.2.1, (σ.2.2.2.1.1, σ.2.2.2.1.2.1, σ.2.2.2.1.2.2.1, T.setSeeds σ.2.2.2.1.2.2.2 u), σ.2.2.2.2)

variable [CharP F 2]

omit [AddCommGroup B] in
theorem view_eq_viewWith (w : W) (ω : T.Rand) :
    T.view w ω = T.viewWith w ω (fun i => T.ideal (T.seedsOf ω.2.2.2.2.2.2 i)) := rfl

omit [AddCommGroup B] in
theorem viewWith_withSeeds (w : W) (ω : T.Rand) (u : T.Hid → B × Salt) (L : T.Hid → D) :
    T.viewWith w (T.withSeeds ω u) L = T.viewWith w ω L := rfl

omit [AddCommGroup B] in
theorem sim_eq_simWith (w0 : W) (σ : T.SimRand) :
    T.sim w0 σ = T.simWith w0 σ (fun i => T.ideal (T.seedsOf σ.2.2.2.1.2.2.2 i)) := rfl

omit [AddCommGroup B] in
theorem simWith_simWithSeeds (w0 : W) (σ : T.SimRand) (u : T.Hid → B × Salt) (L : T.Hid → D) :
    T.simWith w0 (T.simWithSeeds σ u) L = T.simWith w0 σ L := rfl

variable [AddCommGroup Salt] [AddCommGroup Hi] [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]
  [Fintype Salt] [DecidableEq Salt] [Fintype B] [DecidableEq B] [Fintype Hi] [DecidableEq Hi]

omit [AddCommGroup Hi] [DecidableEq F] [Fintype K] [DecidableEq K] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- **Fresh ideal seeds for the hidden leaves give the prover's own distribution**: its view does not read those seeds
anywhere else. -/
theorem prCoin_viewWith_ideal (w : W) (E : T.View → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → B × Salt) => E (T.viewWith w p.1 fun i => T.ideal (p.2 i)))
      = prCoin (fun ω => E (T.view w ω)) := by
  let Φ : T.Rand × (T.Hid → B × Salt) ≃ T.Rand × (T.Hid → B × Salt) :=
    { toFun := fun p => (T.withSeeds p.1 p.2, T.seedsOf p.1.2.2.2.2.2.2)
      invFun := fun p => (T.withSeeds p.1 p.2, T.seedsOf p.1.2.2.2.2.2.2)
      left_inv := fun p => Prod.ext rfl (T.seedsOf_setSeeds _ _)
      right_inv := fun p => Prod.ext rfl (T.seedsOf_setSeeds _ _) }
  have h := prCoin_congr_equiv Φ (fun p => E (T.viewWith w p.1 fun i => T.ideal (p.2 i)))
  rw [← h, ← prCoin_fst (E := T.Hid → B × Salt) (fun ω => E (T.view w ω))]
  rfl

omit [AddCommGroup Hi] [DecidableEq F] [Fintype K] [DecidableEq K] [DecidableEq Salt] [DecidableEq B] [DecidableEq Hi] in
/-- The same for `S_shvzk`'s dummy run. -/
theorem prCoin_simWith_ideal (w0 : W) (E : T.View → Prop) :
    prCoin (fun p : T.SimRand × (T.Hid → B × Salt) => E (T.simWith w0 p.1 fun i => T.ideal (p.2 i)))
      = prCoin (fun σ => E (T.sim w0 σ)) := by
  let Φ : T.SimRand × (T.Hid → B × Salt) ≃ T.SimRand × (T.Hid → B × Salt) :=
    { toFun := fun p => (T.simWithSeeds p.1 p.2, T.seedsOf p.1.2.2.2.1.2.2.2)
      invFun := fun p => (T.simWithSeeds p.1 p.2, T.seedsOf p.1.2.2.2.1.2.2.2)
      left_inv := fun p => Prod.ext rfl (T.seedsOf_setSeeds _ _)
      right_inv := fun p => Prod.ext rfl (T.seedsOf_setSeeds _ _) }
  have h := prCoin_congr_equiv Φ (fun p => E (T.simWith w0 p.1 fun i => T.ideal (p.2 i)))
  rw [← h, ← prCoin_fst (E := T.Hid → B × Salt) (fun σ => E (T.sim w0 σ))]
  rfl

/-- **Lemma B with real leaves** (`docs/zk-proof-public.md` §4.3, `SD ≤ 2·δ₁·N_hid`), for one table: at a fixed
non-degenerate coin vector, under `table_shvzk`'s hypotheses and hm96 hiding (`Hm96Hiding`, `δ₁` per leaf), every event on
the view has probabilities within `2·|Hid|·δ₁` of each other under the real prover (`viewR`: every leaf the final message
doesn't open is a real hm96 leaf of what it commits) and under `S_shvzk` with real leaves (`simR`). `Hid` is the hidden
leaves, level 0's and both pads trees' unopened ones, so `|Hid|` is at most the paper's `N_hid`, which counts every leaf of
those trees: the bound is at least as strong. The proof moves each side to world `W₁` with `ideal_leaves_swap`, at `δ₁` per
leaf, and there applies `table_shvzk`. -/
theorem table_shvzk_hm96 {C : Type} (M : T.Hm96 C) {δ₁ : ℝ≥0∞} (hT1 : Assumptions.Hm96Hiding M.My M.cs δ₁)
    (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w : W) (hreg : T.sReg w = T.pubReg) (hinner : T.InnerHolds w) (w0 : W) (E : T.View → Prop) :
    prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.viewR w p.1 p.2))
        ≤ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2)) + 2 * Fintype.card T.Hid * δ₁
      ∧ prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.viewR w p.1 p.2)) + 2 * Fintype.card T.Hid * δ₁ := by
  have : Nonempty Salt := ⟨0⟩
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have hS := (T.table_shvzk hrank hpad hW hpads hβ w hreg hinner w0).prCoin_eq E
  have hv : prCoin (fun p : T.Rand × (T.Hid → B × Salt) =>
        E (T.viewWith w p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2)))
      = prCoin (fun ω => E (T.view w ω)) := by
    rw [← T.prCoin_viewWith_ideal w E]
    congr 1
    funext p
    congr 2
    funext i
    exact (M.ideal_eq (p.2 i).1 (p.2 i).2).symm
  have hs : prCoin (fun p : T.SimRand × (T.Hid → B × Salt) =>
        E (T.simWith w0 p.1 fun i => M.H ((p.2 i).1, M.cs (p.2 i).2)))
      = prCoin (fun σ => E (T.sim w0 σ)) := by
    rw [← T.prCoin_simWith_ideal w0 E]
    congr 1
    funext p
    congr 2
    funext i
    exact (M.ideal_eq (p.2 i).1 (p.2 i).2).symm
  have v1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.dig w) (fun ω L => E (T.viewWith w ω L))
  have v2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.dig w) (fun ω L => E (T.viewWith w ω L))
  have s1 := ideal_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  have s2 := real_leaves_swap_fintype M.My M.cs hT1 M.H (M.digS w0) (fun σ L => E (T.simWith w0 σ L))
  beta_reduce at v1 v2 s1 s2
  rw [hv] at v1 v2
  rw [hs] at s1 s2
  constructor
  · calc prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.viewR w p.1 p.2))
        ≤ prCoin (fun ω => E (T.view w ω)) + Fintype.card T.Hid * δ₁ := v1
      _ = prCoin (fun σ => E (T.sim w0 σ)) + Fintype.card T.Hid * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2)) + Fintype.card T.Hid * δ₁)
          + Fintype.card T.Hid * δ₁ := add_le_add s2 le_rfl
      _ = _ := by ring
  · calc prCoin (fun p : T.SimRand × (T.Hid → Salt) => E (M.simR w0 p.1 p.2))
        ≤ prCoin (fun σ => E (T.sim w0 σ)) + Fintype.card T.Hid * δ₁ := s1
      _ = prCoin (fun ω => E (T.view w ω)) + Fintype.card T.Hid * δ₁ := by rw [hS]
      _ ≤ (prCoin (fun p : T.Rand × (T.Hid → Salt) => E (M.viewR w p.1 p.2)) + Fintype.card T.Hid * δ₁)
          + Fintype.card T.Hid * δ₁ := add_le_add v2 le_rfl
      _ = _ := by ring

end Table

end FlockSoundness.ZK
