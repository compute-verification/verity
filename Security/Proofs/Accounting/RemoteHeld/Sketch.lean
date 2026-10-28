import Mathlib
import Definitions.Accounting.Guarantees.RemoteSketch
import Proofs.Accounting.RateAudit.DeployedSketchSum

/-!
# The receipts' sketch bound, derived: the proof

`SketchBound` fails on an accepting outcome only where some credited pass's correct receipt is of a block no tick of
the pass had in node memory or on a chip. By `Receipted` that receipt is the kept candidate's correct tag, and by
`Present` that candidate's device did not obtain the block whole: one of its sectors it never held, read, took in or
rebuilt (`Miss`). So `SketchBound`'s failure is within some pass's some candidate's `Miss` at the outcome's model coins
(`miss_of_fails`), an event of the model's coins.

The random-oracle step is `pr_coins_le`: by `SketchHashRandom A ε` the keys SHA-512 gives at every prover coin and
store are within `ε` of uniform, so averaged over those (`pr_le_add_of_fst`) an event of the model's coins is at most
`ε` likelier at the outcome's model coins than over the model's coins. The rest is over the model's coins.

A candidate's `Miss` (`miss_split`) is the verbatim rate audit's guess, at the candidate's own key:

* its signs have a sector block below rank `r`: `ε` per candidate (`SketchRank`);
* the block's tag is right without its query: `N/2^h` per candidate (`Device.pr_guess_le`);
* a query named the block's sketch with a sector never obtained: at rank `r` that sector's `256^(32−r)` values that
  give the sketch are a `2^-(8r)` share of its `2^256`, once per query of the pass's budget (`card_bad_keys_le`,
  `card_bad_family_le` with each candidate's own signs).

One pass's candidates cost `G_j·(ε + N/2^h) + qtot_j/2^(8r)` (`pr_pass_le`), at every prover coin and earlier keys:
pass `j`'s keys and the store are a uniform fiber of the session's coins (`Session.pr_pass_le`). Summed over the passes,
`SketchBoundSession`; at the deployed receipts, `SketchBoundDeployed`.
-/

namespace Accounting.RateSketch.Device

open scoped ENNReal

variable {N m ρ h S L q w : ℕ} {X Sig Dg : Type}

/-- **The budgeted union with each candidate's own signs.** `card_bad_family_le` for a family whose candidates differ
in their signs too: if across the family the queries total at most `qtot` on every data point, the data on which some
candidate makes a correct digest query without an obtained sector is at most a `qtot·g/|X|` fraction. -/
theorem card_bad_keys_le [Fintype X] [DecidableEq X] [Fintype Dg] [DecidableEq Dg] {C : ℕ}
    (A : Fin C → Device N m ρ h S L q w X Sig Dg) (dig : Sig → (Fin m → X) → Dg) (σ : Fin C → Sig)
    (O : Fin C → Oracle N ρ h X Dg) (g qtot : ℕ)
    (hq : ∀ D : Data N m ρ X, ∑ c, (((A c).prog (σ c) D).queries (O c) D).length ≤ qtot)
    [DecidablePred (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig (σ c) (O c) D i g)] :
    (Finset.univ.filter (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig (σ c) (O c) D i g)).card *
        Fintype.card X ≤ g * qtot * Fintype.card (Data N m ρ X) := by
  classical
  have hsub : (Finset.univ.filter
      (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig (σ c) (O c) D i g)).card ≤
      ∑ ci : Fin C × Fin q,
        (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig (σ ci.1) (O ci.1) D ci.2 g)).card := by
    refine (Finset.card_le_card (fun D hD => ?_)).trans Finset.card_biUnion_le
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hD
    obtain ⟨c, i, hci⟩ := hD
    exact Finset.mem_biUnion.2 ⟨(c, i), Finset.mem_univ _,
      Finset.mem_filter.2 ⟨Finset.mem_univ _, hci⟩⟩
  have hcount : ∀ c : Fin C, ∑ i : Fin q, (Finset.univ.filter
      (fun D : Data N m ρ X => (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length)).card ≤
      ∑ D : Data N m ρ X, (((A c).prog (σ c) D).queries (O c) D).length := by
    intro c
    calc (∑ i : Fin q, (Finset.univ.filter
            (fun D : Data N m ρ X => (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length)).card)
        = ∑ i : Fin q, ∑ D : Data N m ρ X,
            if (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length then 1 else 0 :=
          Finset.sum_congr rfl fun i _ => by rw [Finset.card_filter]
      _ = ∑ D : Data N m ρ X, ∑ i : Fin q,
            if (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length then 1 else 0 :=
          Finset.sum_comm
      _ ≤ ∑ D : Data N m ρ X, (((A c).prog (σ c) D).queries (O c) D).length := by
          refine Finset.sum_le_sum fun D _ => ?_
          calc (∑ i : Fin q,
                  if (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length then 1 else 0)
              = (Finset.univ.filter
                  (fun i : Fin q => (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length)).card :=
                (Finset.card_filter _ _).symm
            _ ≤ (Finset.range ((((A c).prog (σ c) D).queries (O c) D).length)).card := by
                refine Finset.card_le_card_of_injOn (fun i => (i : ℕ)) (fun i hi => ?_)
                  (Fin.val_injective.injOn)
                simp only [Finset.coe_filter, Finset.mem_univ, true_and, Set.mem_ofPred_eq] at hi
                simpa using hi
            _ = (((A c).prog (σ c) D).queries (O c) D).length := Finset.card_range _
  calc (Finset.univ.filter
          (fun D : Data N m ρ X => ∃ c, ∃ i : Fin q, Bad (A c) dig (σ c) (O c) D i g)).card * Fintype.card X
      ≤ (∑ ci : Fin C × Fin q,
          (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig (σ ci.1) (O ci.1) D ci.2 g)).card) *
            Fintype.card X := Nat.mul_le_mul_right _ hsub
    _ = ∑ ci : Fin C × Fin q,
          (Finset.univ.filter (fun D : Data N m ρ X => Bad (A ci.1) dig (σ ci.1) (O ci.1) D ci.2 g)).card *
            Fintype.card X := by rw [Finset.sum_mul]
    _ ≤ ∑ ci : Fin C × Fin q, g * (Finset.univ.filter
          (fun D : Data N m ρ X =>
            (ci.2 : ℕ) < (((A ci.1).prog (σ ci.1) D).queries (O ci.1) D).length)).card :=
        Finset.sum_le_sum fun ci _ => card_bad_ground_le (A ci.1) dig (σ ci.1) (O ci.1) ci.2 g
    _ = g * ∑ c, ∑ i : Fin q, (Finset.univ.filter
          (fun D : Data N m ρ X => (i : ℕ) < (((A c).prog (σ c) D).queries (O c) D).length)).card := by
        rw [← Finset.mul_sum, Fintype.sum_prod_type]
    _ ≤ g * ∑ c, ∑ D : Data N m ρ X, (((A c).prog (σ c) D).queries (O c) D).length :=
        Nat.mul_le_mul_left g (Finset.sum_le_sum fun c _ => hcount c)
    _ = g * ∑ D : Data N m ρ X, ∑ c, (((A c).prog (σ c) D).queries (O c) D).length := by
        rw [Finset.sum_comm]
    _ ≤ g * ∑ _D : Data N m ρ X, qtot :=
        Nat.mul_le_mul_left g (Finset.sum_le_sum fun D _ => hq D)
    _ = g * qtot * Fintype.card (Data N m ρ X) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_assoc, mul_comm qtot]

end Accounting.RateSketch.Device

namespace Accounting.Remote.Sketch

open Accounting.RateSketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_mono pr_or_le)

variable {N m h S L q ks : ℕ}

/-- A device's correct receipt of a block one of whose sectors it never obtained, at a key. -/
def Miss (A : RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (κ : Key N m h ks) (D : Data N m 0 Sector) : Prop :=
  ∃ t ∈ A.correct sketch κ.1 κ.2 D, ∃ u, Sum.inl (t, u) ∉ A.obtained κ.1 κ.2 D

/-- Some sector block of the signs is below rank `r`. -/
def LowRank (r : ℕ) (σ : SketchSigns ks m) : Prop :=
  ∃ u, ((Accounting.Tag.signMatrix (σ.block u)).map (Int.cast : ℤ → ℚ)).rank < r

/-- A correct receipt of a block with a sector never obtained: its tag was never queried, or a query named the block's
sketch with that sector unobtained. -/
theorem miss_split
    (A : RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (σ : SketchSigns ks m) (O : Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) (D : Data N m 0 Sector) (g : ℕ)
    (hfib : fib (sketch (k := ks) (m := m)) σ ≤ g) (t : Fin N) (u : Fin m)
    (hu : Sum.inl (t, u) ∉ A.obtained σ O D) :
    t ∉ queried A sketch σ O D ∨ ∃ i : Fin q, Bad A sketch σ O D i g := by
  classical
  by_cases hq : t ∈ queried A sketch σ O D
  · right
    have hqmem : (D.stream, t, sketch σ (D.tile t)) ∈ (A.prog σ D).queries O D := by
      simpa [queried] using hq
    have hlt := List.idxOf_lt_length_of_mem hqmem
    have hget : ((A.prog σ D).queries O D)[
        ((A.prog σ D).queries O D).idxOf (D.stream, t, sketch σ (D.tile t))]? =
        some (D.stream, t, sketch σ (D.tile t)) := by
      rw [List.getElem?_eq_getElem hlt, List.getElem_idxOf hlt]
    refine ⟨⟨_, hlt.trans_le (A.bounded σ _ O D).2⟩, hfib, ⟨Sum.inl (t, u), ?_⟩, _, hget, rfl⟩
    simp only [unobtained, Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hu, _, hget, rfl⟩
  · exact Or.inl hq

/-- The rank tail at one candidate's key, over the store and the key. -/
theorem pr_lowRank_le (r : ℕ) (ε : ℝ≥0∞) (hRank : Accounting.Assumptions.SketchRank ks m r ε) :
    pr (fun z : Data N m 0 Sector × Key N m h ks => LowRank r z.2.1) ≤ ε := by
  let _ : Fintype (Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  let e : Data N m 0 Sector × Key N m h ks ≃
      SketchSigns ks m × (Data N m 0 Sector × Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) :=
    { toFun := fun z => (z.2.1, (z.1, z.2.2))
      invFun := fun y => (y.2.1, (y.1, y.2.2))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [Accounting.Rate.pr_equiv e]
  exact Accounting.Rate.pr_le_of_fiber _ _ (fun _ => hRank)

/-- The tag guess at one candidate's key: `1/2^h` per block, over the store and the key. -/
theorem pr_guess_key_le
    (A : RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32))) (t : Fin N) :
    pr (fun z : Data N m 0 Sector × Key N m h ks =>
      t ∈ A.correct sketch z.2.1 z.2.2 z.1 ∧ t ∉ queried A sketch z.2.1 z.2.2 z.1) ≤ (1 : ℝ≥0∞) / 2 ^ h := by
  let _ : Fintype (Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) := inferInstance
  let e : Data N m 0 Sector × Key N m h ks ≃
      (Data N m 0 Sector × Oracle N 0 h Sector (Fin ks → ZMod (2 ^ 32))) × SketchSigns ks m :=
    { toFun := fun z => ((z.1, z.2.2), z.2.1)
      invFun := fun y => (y.1.1, (y.2, y.1.2))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [Accounting.Rate.pr_equiv e]
  exact Accounting.Rate.pr_le_of_fiber _ _ (fun σ => Device.pr_guess_le A σ t)

/-- **One pass, all its candidates.** For a fixed family of `C` candidate devices on the store, each at its own uniform
key, within the pass's budget of `qtot` queries: some candidate misses with probability at most
`C·(ε + N/2^h) + qtot/2^(8r)`. -/
theorem pr_pass_le {C : ℕ}
    (A : Fin C → RateSketch.Device N m 0 h S L q 256 Sector (SketchSigns ks m) (Fin ks → ZMod (2 ^ 32)))
    (r qtot : ℕ) (ε : ℝ≥0∞) (hr32 : r ≤ 32) (hRank : Accounting.Assumptions.SketchRank ks m r ε)
    (hq : ∀ (κ : Fin C → Key N m h ks) (D : Data N m 0 Sector),
      ∑ c, (((A c).prog (κ c).1 D).queries (κ c).2 D).length ≤ qtot) :
    pr (fun x : Data N m 0 Sector × (Fin C → Key N m h ks) => ∃ c, Miss (A c) (x.2 c) x.1)
      ≤ (C : ℝ≥0∞) * (ε + (N : ℝ≥0∞) / 2 ^ h) + (qtot : ℝ≥0∞) / 2 ^ (8 * r) := by
  classical
  let g := 256 ^ (32 - r)
  have hsplit : ∀ x : Data N m 0 Sector × (Fin C → Key N m h ks), (∃ c, Miss (A c) (x.2 c) x.1) →
      ((∃ c, LowRank r (x.2 c).1) ∨ ∃ c, ∃ t, t ∈ (A c).correct sketch (x.2 c).1 (x.2 c).2 x.1 ∧
          t ∉ queried (A c) sketch (x.2 c).1 (x.2 c).2 x.1) ∨
        ∃ c, ∃ i : Fin q, Bad (A c) sketch (x.2 c).1 (x.2 c).2 x.1 i g := by
    rintro ⟨D, κ⟩ ⟨c, t, ht, u, hu⟩
    by_cases hl : LowRank r (κ c).1
    · exact Or.inl (Or.inl ⟨c, hl⟩)
    · have hr : ∀ u, r ≤ ((Accounting.Tag.signMatrix ((κ c).1.block u)).map (Int.cast : ℤ → ℚ)).rank := by
        intro u
        by_contra hlt
        exact hl ⟨u, by omega⟩
      have hfib := Accounting.SecurityProofs.SketchRankFiber ks m r (κ c).1 hr
      rcases miss_split (A c) (κ c).1 (κ c).2 D g hfib t u hu with hg | hb
      · exact Or.inl (Or.inr ⟨c, t, ht, hg⟩)
      · exact Or.inr ⟨c, hb⟩
  refine (pr_mono hsplit).trans ((pr_or_le _ _).trans (add_le_add ((pr_or_le _ _).trans ?_) ?_))
  · rw [mul_add]
    refine add_le_add ?_ ?_
    · refine (pr_exists_le _).trans ?_
      refine (Finset.sum_le_sum (fun c _ => Device.pr_component_le c
        (fun z : Data N m 0 Sector × Key N m h ks => LowRank r z.2.1) ε (pr_lowRank_le r ε hRank))).trans
        (le_of_eq ?_)
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    · refine (pr_exists_le _).trans ?_
      have hc : ∀ c : Fin C,
          pr (fun x : Data N m 0 Sector × (Fin C → Key N m h ks) =>
            ∃ t, t ∈ (A c).correct sketch (x.2 c).1 (x.2 c).2 x.1 ∧
              t ∉ queried (A c) sketch (x.2 c).1 (x.2 c).2 x.1) ≤ (N : ℝ≥0∞) / 2 ^ h := by
        intro c
        refine (pr_exists_le _).trans ?_
        have ht : ∀ t : Fin N,
            pr (fun x : Data N m 0 Sector × (Fin C → Key N m h ks) =>
              t ∈ (A c).correct sketch (x.2 c).1 (x.2 c).2 x.1 ∧
                t ∉ queried (A c) sketch (x.2 c).1 (x.2 c).2 x.1) ≤ (1 : ℝ≥0∞) / 2 ^ h := fun t =>
          Device.pr_component_le c
            (fun z : Data N m 0 Sector × Key N m h ks =>
              t ∈ (A c).correct sketch z.2.1 z.2.2 z.1 ∧ t ∉ queried (A c) sketch z.2.1 z.2.2 z.1)
            _ (pr_guess_key_le (A c) t)
        refine (Finset.sum_le_sum (fun t _ => ht t)).trans (le_of_eq ?_)
        simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, div_eq_mul_inv]
      refine (Finset.sum_le_sum (fun c _ => hc c)).trans (le_of_eq ?_)
      simp [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  · refine Accounting.Rate.pr_le_of_fiber _ _ (fun κ => ?_)
    have hcard := Device.card_bad_keys_le A sketch (fun c => (κ c).1) (fun c => (κ c).2) g qtot
      (fun D => hq κ D)
    have hbound := Bandwidth.pr_le_of_card
      (fun D : Data N m 0 Sector => ∃ c, ∃ i : Fin q, Bad (A c) sketch (κ c).1 (κ c).2 D i g)
      (Fintype.card Sector) (g * qtot) (by simp [Sector]) (by
        simpa using hcard)
    have hXc : (Fintype.card Sector : ℝ≥0∞) = 2 ^ 256 := by
      norm_num [Sector]
    rw [hXc] at hbound
    refine hbound.trans (le_of_eq ?_)
    have hg : ((g : ℕ) : ℝ≥0∞) = 2 ^ (256 - 8 * r) := by
      dsimp [g]
      push_cast
      rw [show (256 : ℝ≥0∞) = 2 ^ 8 by norm_num, ← pow_mul,
        show 8 * (32 - r) = 256 - 8 * r by omega]
    have hsplitPow : (2 : ℝ≥0∞) ^ 256 = 2 ^ (256 - 8 * r) * 2 ^ (8 * r) := by
      rw [← pow_add, Nat.sub_add_cancel (by omega)]
    push_cast
    rw [hg, hsplitPow, mul_comm ((2 : ℝ≥0∞) ^ (256 - 8 * r)) ((qtot : ℝ≥0∞))]
    rw [show (qtot : ℝ≥0∞) * 2 ^ (256 - 8 * r) / (2 ^ (256 - 8 * r) * 2 ^ (8 * r)) =
        2 ^ (256 - 8 * r) * (qtot : ℝ≥0∞) / (2 ^ (256 - 8 * r) * 2 ^ (8 * r)) by ring_nf]
    exact ENNReal.mul_div_mul_left _ _ (by positivity) (ENNReal.pow_ne_top ENNReal.ofNat_ne_top)

/-- A probability over a pair is the average over the first coordinate of its fibers' probabilities. -/
theorem pr_eq_sum_div_fst {X Y : Type} [Fintype X] [Fintype Y] (Q : X × Y → Prop) :
    pr Q = (∑ x, pr fun y => Q (x, y)) / Fintype.card X := by
  classical
  unfold pr
  have e : (Finset.univ.filter Q).card = ∑ x, (Finset.univ.filter fun y => Q (x, y)).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type]
    simp only [Finset.card_filter]
  rw [e, Nat.cast_sum, Fintype.card_prod, Nat.cast_mul, ← ENNReal.sum_div]
  simp only [ENNReal.div_eq_inv_mul]
  rw [ENNReal.mul_inv (Or.inr (ENNReal.natCast_ne_top _)) (Or.inl (ENNReal.natCast_ne_top _))]
  ring

/-- **Averaging.** A bound that holds at every value of the first coordinate, up to `e`, holds for the pair. -/
theorem pr_le_add_of_fst {X Y Z : Type} [Fintype X] [Fintype Y] [Fintype Z] (P : X × Y → Prop)
    (Q : X × Z → Prop) (e : ℝ≥0∞) (h : ∀ x, pr (fun y => P (x, y)) ≤ pr (fun z => Q (x, z)) + e) :
    pr P ≤ pr Q + e := by
  rw [pr_eq_sum_div_fst P, pr_eq_sum_div_fst Q]
  calc (∑ x, pr fun y => P (x, y)) / Fintype.card X
      ≤ (∑ x, (pr (fun z => Q (x, z)) + e)) / Fintype.card X :=
        ENNReal.div_le_div_right (Finset.sum_le_sum fun x _ => h x) _
    _ = (∑ x, pr fun z => Q (x, z)) / Fintype.card X + Fintype.card X * e / Fintype.card X := by
        rw [Finset.sum_add_distrib, ENNReal.add_div, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    _ ≤ _ := add_le_add le_rfl (ENNReal.div_le_of_le_mul (mul_comm _ _).le)

namespace Session

variable {β Hs : Type} [Fintype β] {P : ℕ} {G : Fin P → ℕ}

/-- **The random-oracle step.** With SHA-512's keys at the candidate seeds within `ε` of uniform at every prover coin
and store (`SketchHashRandom A ε`), an event of the model's coins is at most `ε` likelier at the outcome's model coins
than over the model's coins: average the premise over the prover's coins and the store. -/
theorem pr_coins_le [Fintype Hs] (A : Session β Hs P N m h S L q ks G) {ε : ℝ}
    (hH : Accounting.Assumptions.SketchHashRandom A ε) (E : ModelCoins β P N m h ks G → Prop) :
    pr (fun ω : Outcome β Hs N m => E (A.coins ω)) ≤ pr E + ENNReal.ofReal ε := by
  refine pr_le_add_of_fst (fun ω : β × (Data N m 0 Sector × Hs) => E (A.coins ω)) E _ fun b => ?_
  refine pr_le_add_of_fst (fun y : Data N m 0 Sector × Hs => E (A.coins (b, y))) (fun z => E (b, z)) _
    fun D => ?_
  exact hH b D fun κ => E (b, D, κ)

/-- **Sequencing.** Pass `j`'s candidates miss with the one-pass bound for every device family the prover's coins and
the earlier passes' keys select: pass `j`'s keys and the store are a uniform fiber of the model's coins. -/
theorem pr_pass_le (A : Session β Hs P N m h S L q ks G) (j : Fin P) (δ : ℝ≥0∞)
    (hδ : ∀ b η, pr (fun y : Data N m 0 Sector × (Fin (G j) → Key N m h ks) =>
      ∃ c, Miss (A.device b j η c) (y.2 c) y.1) ≤ δ) :
    pr (fun ω : ModelCoins β P N m h ks G => ∃ c, Miss (A.devAt ω j c) (ω.2.2 j c) ω.2.1) ≤ δ := by
  classical
  let sp := Equiv.piSplitAt j (fun i : Fin P => Fin (G i) → Key N m h ks)
  let shuffle : β × (Data N m 0 Sector × ((Fin (G j) → Key N m h ks) ×
        ((i : {i : Fin P // i ≠ j}) → Fin (G i) → Key N m h ks))) ≃
      (Data N m 0 Sector × (Fin (G j) → Key N m h ks)) ×
        (β × ((i : {i : Fin P // i ≠ j}) → Fin (G i) → Key N m h ks)) :=
    { toFun := fun x => ((x.2.1, x.2.2.1), (x.1, x.2.2.2))
      invFun := fun y => (y.2.1, (y.1.1, (y.1.2, y.2.2)))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  let e : ModelCoins β P N m h ks G ≃ (Data N m 0 Sector × (Fin (G j) → Key N m h ks)) ×
        (β × ((i : {i : Fin P // i ≠ j}) → Fin (G i) → Key N m h ks)) :=
    ((Equiv.refl β).prodCongr ((Equiv.refl _).prodCongr sp)).trans shuffle
  rw [Accounting.Rate.pr_equiv e]
  refine Accounting.Rate.pr_le_of_fiber _ _ (fun rest => ?_)
  let η : (i : {i : Fin P // i < j}) → Fin (G i) → Key N m h ks := fun i => rest.2 ⟨i.1, Fin.ne_of_lt i.2⟩
  refine (pr_mono (fun x hx => ?_)).trans (hδ rest.1 η)
  have h1 : (e.symm (x, rest)).1 = rest.1 := rfl
  have h2 : (e.symm (x, rest)).2.1 = x.1 := rfl
  have h3 : (e.symm (x, rest)).2.2 = sp.symm (x.2, rest.2) := rfl
  have h4 : sp.symm (x.2, rest.2) j = x.2 := by simp [sp]
  have h5 : (fun i : {i : Fin P // i < j} => sp.symm (x.2, rest.2) i.1) = η := by
    funext i
    simp [sp, η, Fin.ne_of_lt i.2]
  obtain ⟨c, hc⟩ := hx
  refine ⟨c, ?_⟩
  simp only [devAt, h1, h2, h3] at hc
  rw [h5, h4] at hc
  exact hc

end Session

/-- An accepting outcome where `SketchBound`'s event fails has some pass's kept candidate missing at the outcome's
model coins. -/
theorem miss_of_fails {β Hs : Type} {P : ℕ} {G : Fin P → ℕ} (A : Session β Hs P N m h S L q ks G)
    (T : Accounting.Remote.Held.Trace (Outcome β Hs N m)) (hP : T.passes ≤ P)
    (hrec : Assumptions.Receipted A T) (hpres : Assumptions.Present A T) (ω : Outcome β Hs N m)
    (hacc : T.accepts ω)
    (hfail : ¬ ∀ p < T.passes, ∀ b ∈ T.correct ω p,
      ∃ τ, T.tick ω p ≤ τ ∧ τ < T.tick ω (p + 1) ∧ b ∈ T.present ω τ) :
    ∃ j : Fin P, ∃ c, Miss (A.devAt (A.coins ω) j c) ((A.coins ω).2.2 j c) (A.coins ω).2.1 := by
  push Not at hfail
  obtain ⟨p, hp, b, hb, hnot⟩ := hfail
  let j : Fin P := ⟨p, lt_of_lt_of_le hp hP⟩
  obtain ⟨t, ht, htb⟩ := hrec ω hacc j b hb
  refine ⟨j, A.kept ω j, t, ht, ?_⟩
  by_contra hall
  push Not at hall
  obtain ⟨τ, h1, h2, h3⟩ := hpres ω hacc j t hall b htb.symm
  exact absurd h3 (hnot τ h1 h2)

end Accounting.Remote.Sketch

namespace Accounting.SecurityProofs.Remote

open Accounting.RateSketch Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Sketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_exists_le pr_mono)

/-- `Pouw.pr` is monotone (`Accounting.PouwWork.pr_mono`, restated to keep this file's imports). -/
private theorem pouw_pr_mono {α : Type} [Fintype α] {E F : α → Prop} (h : ∀ x, E x → F x) :
    Pouw.pr E ≤ Pouw.pr F := by
  classical
  unfold Pouw.pr
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hsub : (Finset.univ.filter E) ⊆ Finset.univ.filter F := fun x hx => by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx ⊢
    exact h x hx
  exact_mod_cast Finset.card_le_card hsub

/-- **The sketch bound at every parameter** (`Accounting.Guarantees.Remote.SketchBoundSession`). -/
theorem SketchBoundSession : Accounting.Guarantees.Remote.SketchBoundSession := by
  intro β Hs _ _ P N m h S L q ks r G qtot ε εro A T hr32 hε hεro hP hRank hH hq hrec hpres
  classical
  let δ : ℝ≥0∞ := ((∑ j, G j : ℕ) : ℝ≥0∞) * (ENNReal.ofReal ε + (N : ℝ≥0∞) / 2 ^ h) +
    ((∑ j, qtot j : ℕ) : ℝ≥0∞) / 2 ^ (8 * r)
  have hsum : pr (fun x : ModelCoins β P N m h ks G => ∃ j : Fin P, ∃ c, Miss (A.devAt x j c) (x.2.2 j c) x.2.1) ≤ δ := by
    refine (pr_exists_le _).trans ?_
    calc ∑ j : Fin P, pr (fun x : ModelCoins β P N m h ks G => ∃ c, Miss (A.devAt x j c) (x.2.2 j c) x.2.1)
        ≤ ∑ j : Fin P, ((G j : ℝ≥0∞) * (ENNReal.ofReal ε + (N : ℝ≥0∞) / 2 ^ h) +
            (qtot j : ℝ≥0∞) / 2 ^ (8 * r)) :=
          Finset.sum_le_sum fun j _ => Session.pr_pass_le A j _ fun b η =>
            pr_pass_le (fun c => A.device b j η c) r (qtot j) (ENNReal.ofReal ε) hr32 hRank
              (fun κ D => hq b j η κ D)
      _ = δ := by
          simp only [δ, Finset.sum_add_distrib, ← Finset.sum_mul, ENNReal.sum_div, Nat.cast_sum]
  have hout : pr (fun ω : Outcome β Hs N m =>
      ∃ j : Fin P, ∃ c, Miss (A.devAt (A.coins ω) j c) ((A.coins ω).2.2 j c) (A.coins ω).2.1) ≤
      δ + ENNReal.ofReal εro :=
    (Session.pr_coins_le A hH
      (fun x : ModelCoins β P N m h ks G => ∃ j : Fin P, ∃ c, Miss (A.devAt x j c) (x.2.2 j c) x.2.1)).trans
      (add_le_add hsum le_rfl)
  have hfin : δ ≠ ⊤ := by
    simp only [δ]
    refine ENNReal.add_ne_top.2 ⟨ENNReal.mul_ne_top (ENNReal.natCast_ne_top _)
      (ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top, ?_⟩), ?_⟩
    · exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by positivity)
    · exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by positivity)
  have hδ : δ.toReal = ((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r) := by
    simp only [δ]
    rw [ENNReal.toReal_add, ENNReal.toReal_mul, ENNReal.toReal_add, ENNReal.toReal_ofReal hε,
      ENNReal.toReal_div, ENNReal.toReal_div]
    · simp only [ENNReal.toReal_natCast, ENNReal.toReal_pow, ENNReal.toReal_ofNat]
    · exact ENNReal.ofReal_ne_top
    · exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by positivity)
    · exact ENNReal.mul_ne_top (ENNReal.natCast_ne_top _)
        (ENNReal.add_ne_top.2 ⟨ENNReal.ofReal_ne_top,
          ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by positivity)⟩)
    · exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by positivity)
  have hδ' : (δ + ENNReal.ofReal εro).toReal =
      ((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ h) + ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ (8 * r) + εro := by
    rw [ENNReal.toReal_add hfin ENNReal.ofReal_ne_top, hδ, ENNReal.toReal_ofReal hεro]
  have hmono : Pouw.pr (fun ω => T.accepts ω ∧ ¬ ∀ p < T.passes, ∀ b ∈ T.correct ω p,
      ∃ τ, T.tick ω p ≤ τ ∧ τ < T.tick ω (p + 1) ∧ b ∈ T.present ω τ) ≤
      Pouw.pr (fun ω : Outcome β Hs N m =>
        ∃ j : Fin P, ∃ c, Miss (A.devAt (A.coins ω) j c) ((A.coins ω).2.2 j c) (A.coins ω).2.1) :=
    pouw_pr_mono fun ω hω => miss_of_fails A T hP hrec hpres ω hω.1 hω.2
  have hpr : Pouw.pr (fun ω : Outcome β Hs N m =>
        ∃ j : Fin P, ∃ c, Miss (A.devAt (A.coins ω) j c) ((A.coins ω).2.2 j c) (A.coins ω).2.1) =
      (pr (fun ω : Outcome β Hs N m =>
        ∃ j : Fin P, ∃ c, Miss (A.devAt (A.coins ω) j c) ((A.coins ω).2.2 j c) (A.coins ω).2.1)).toReal := by
    unfold Pouw.pr Pous.pr
    rw [ENNReal.toReal_div]
    simp
  unfold SketchBound Trace.Holds
  refine hmono.trans ?_
  rw [hpr, ← hδ']
  exact ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨hfin, ENNReal.ofReal_ne_top⟩) hout

/-- **The sketch bound at the deployed receipts** (`Accounting.Guarantees.Remote.SketchBoundDeployed`):
`SketchBoundSession` at `m = 128`, 16 rows, `h = 512`, `r = 12`, with at most `Q + P` candidate seeds and `Q` queries,
at `ε_ro = (Q + P)²/2^257`. -/
theorem SketchBoundDeployed : Accounting.Guarantees.Remote.SketchBoundDeployed := by
  intro β Hs _ _ P N S L q Q G qtot ε A T hε hP hRank hG hqtot hH hq hrec hpres
  have h := SketchBoundSession β Hs P N 128 512 S L q 16 12 G qtot ε (((Q : ℝ) + P) ^ 2 / 2 ^ 257) A T (by norm_num)
    hε (by positivity) hP hRank hH hq hrec hpres
  unfold SketchBound Trace.Holds at h ⊢
  rw [show (8 * 12 : ℕ) = 96 from rfl] at h
  refine h.trans ?_
  have hG' : ((∑ j, G j : ℕ) : ℝ) ≤ (Q : ℝ) + P := by exact_mod_cast hG
  have hq' : ((∑ j, qtot j : ℕ) : ℝ) ≤ Q := by exact_mod_cast hqtot
  have hnn : (0 : ℝ) ≤ ε + N / 2 ^ 512 := add_nonneg hε (by positivity)
  have h1 : ((∑ j, G j : ℕ) : ℝ) * (ε + N / 2 ^ 512) ≤ ((Q : ℝ) + P) * (ε + N / 2 ^ 512) :=
    mul_le_mul_of_nonneg_right hG' hnn
  have h2 : ((∑ j, qtot j : ℕ) : ℝ) / 2 ^ 96 ≤ (Q : ℝ) / 2 ^ 96 := by gcongr
  exact (add_le_add (add_le_add h1 h2) le_rfl).trans (le_of_eq (by ring))

end Accounting.SecurityProofs.Remote
