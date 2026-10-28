import Mathlib
import Definitions.Pouw.Guarantees.RelativeSecurity
import Proofs.Pouw.RelativeSecurity

/-!
# Relative security's fork finder: the floors per prover's error, and their honest and nontrivial lemmas

The floors per prover conclude at `breakPr`, each adversary's error at its fork finders' collision probabilities. This
file proves the step every one of them takes, `serviceFloorPerProver_breakPr_of_slices` (the window's slices averaged,
each capped at 1), and the reading of that error through `ecr/sha-512` at budgets computed from the adversary,
`breakPr_le_breakBound`.

It also proves the honest lemmas of the six floors per prover, on one round over one cell-block in the machine of
functions at no cost: the honest run's one product slot computes the cell-block's target, so every error table is
zero, the game accepts every play and no round breaks; and their nontrivial lemma, a constant adversary against a
verifier that accepts everything.
-/

namespace Pouw.RelativeSecurity

open Finset

/-- **The service floor per prover at the fork finders' collision probabilities**: as
`serviceFloorPerProver_of_slices`, with each adversary's error read at its finders' collision probabilities
(`breakPr`) rather than through `ecr/sha-512`: each slice's bound is at most 1 and at most
`ε + R (δb + δz) + βb Σ_w Pr[fb … w] + βz Σ_w Pr[fz … w]`, and the slices average. -/
theorem serviceFloorPerProver_breakPr_of_slices {Mc : Machine} {S : Service} {Ω : Type} [Fintype Ω] {R : ℕ}
    {p F₀ sl Vmax δb βb δz βz : ℝ} (e : S.Seed ≃ Paths Ω R) (sha : Claims.Msg → Claims.Msg) (fb fz : Finders Mc S R)
    (h : ∀ (Ω' : Type) [Fintype Ω'] (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)),
      (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) → ∀ r : Ω',
      ∃ Bind ZK : Fin R → Paths Ω R → Prop,
        (∀ w, prob (Bind w) ≤ δb + βb * (fb Ω' M r w).Pr sha) ∧ (∀ w, prob (ZK w) ≤ δz + βz * (fz Ω' M r w).Pr sha) ∧
        RoundsSlice p F₀ sl Vmax
          (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
          (fun x => Mc.charge M (e.symm x, r)) Bind ZK)
    (hp : p ≤ 1) (hF₀ : 0 < F₀) {ε : ℝ} (D : ℝ) (hε : Real.exp (-((1 - p) * D)) ≤ ε) :
    ServiceFloorPerProver Mc S ((R - D) * F₀ - Vmax - R * sl) (breakPr sha fb fz ε δb βb δz βz) := by
  intro Ω' _ M hc
  have hc0 : (0 : ℝ) ≤ (Fintype.card Ω' : ℝ)⁻¹ := inv_nonneg.mpr (Nat.cast_nonneg _)
  have hinv : ENNReal.ofReal (Fintype.card Ω' : ℝ)⁻¹ ≤ (Fintype.card Ω' : ENNReal)⁻¹ := by
    rcases Nat.eq_zero_or_pos (Fintype.card Ω') with h0 | h0
    · simp [h0]
    · have hpos : (0 : ℝ) < Fintype.card Ω' := by exact_mod_cast h0
      rw [ENNReal.ofReal_inv_of_pos hpos, ENNReal.ofReal_natCast]
  rw [prob_prod_eq_sum, ENNReal.ofReal_sum_of_nonneg fun r _ => ?_]
  swap
  · exact mul_nonneg hc0 (prob_nonneg _)
  unfold breakPr
  refine Finset.sum_le_sum fun r _ => ?_
  rw [ENNReal.ofReal_mul hc0]
  refine mul_le_mul' hinv (le_min (ENNReal.ofReal_le_one.mpr (prob_le_one _)) ?_)
  obtain ⟨Bind, ZK, hB, hZ, hsl⟩ := h Ω' M hc r
  have hs1 : ∑ w : Fin R, (prob (Bind w) + prob (ZK w)) ≤
      ∑ w : Fin R, ((δb + δz) + (βb * (fb Ω' M r w).Pr sha + βz * (fz Ω' M r w).Pr sha)) :=
    Finset.sum_le_sum fun w _ => by linarith [hB w, hZ w]
  have hs2 : ∑ w : Fin R, ((δb + δz) + (βb * (fb Ω' M r w).Pr sha + βz * (fz Ω' M r w).Pr sha)) =
      R * (δb + δz) + ((βb * ∑ w, (fb Ω' M r w).Pr sha) + βz * ∑ w, (fz Ω' M r w).Pr sha) := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.sum_add_distrib, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, ← Finset.mul_sum, ← Finset.mul_sum]
  refine ENNReal.ofReal_le_ofReal ?_
  rw [prob_equiv e]
  refine (hsl.prob_le hp hF₀ D).trans ?_
  linarith [hs1, hs2, hε]

/-- **The fork finders' error read through `ecr/sha-512`**: at any budget constants, an adversary's error at its
finders' collision probabilities is at most its `breakBound`, which reads each finder through `Claims.ecrBound` at
`cb · time M + db` (binding) and `cz · time M + dz` (zero knowledge): `ofReal (Pr) ≤ ecrBound` at every budget. -/
theorem breakPr_le_breakBound {Mc : Machine} {S : Service} {R : ℕ} (sha : Claims.Msg → Claims.Msg)
    (fb fz : Finders Mc S R) (cb db cz dz ε δb βb δz βz : ℝ) (hβb : 0 ≤ βb) (hβz : 0 ≤ βz) (Ω : Type) [Fintype Ω]
    (M : Mc.Alg (S.Seed × Ω) (S.Com × S.Ans)) :
    breakPr sha fb fz ε δb βb δz βz Ω M ≤ breakBound sha fb fz cb db cz dz ε δb βb δz βz Ω M := by
  unfold breakPr breakBound
  refine Finset.sum_le_sum fun r _ => mul_le_mul' le_rfl (min_le_min le_rfl ?_)
  have hfd : ∀ (fd : Finders Mc S R) (q β : ℝ), 0 ≤ β →
      ENNReal.ofReal (β * ∑ w, (fd Ω M r w).Pr sha) ≤ ENNReal.ofReal β * finderTerm sha fd q Ω M r := by
    intro fd q β hβ
    rw [ENNReal.ofReal_mul hβ, ENNReal.ofReal_sum_of_nonneg fun w _ => ?_]
    · unfold finderTerm
      exact mul_le_mul' le_rfl (Finset.sum_le_sum fun w _ => ofReal_finderPr_le_ecrBound _ _ _)
    · unfold Claims.Finder.Pr Claims.Pr
      positivity
  exact (ENNReal.ofReal_add_le.trans (add_le_add ENNReal.ofReal_add_le le_rfl)).trans
    (add_le_add (add_le_add le_rfl (hfd fb _ βb hβb)) (hfd fz _ βz hβz))

end Pouw.RelativeSecurity

namespace Pouw.SecurityProofs.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions Finset

/-! ## The honest instances -/

section Honest

/-- The honest service on seeds `Ω`: one commitment, a Boolean answer, accepted where it is `true`; the verifier hashes
nothing. -/
def honestService (Ω : Type) [Fintype Ω] (lam : ℕ) (h : Fintype.card Ω = 2 ^ lam) : HashedService where
  Seed := Ω
  lam := lam
  card_seed := h
  Com := Unit
  Ans := Bool
  Coins := Unit
  verify _ _ a := a
  hashes _ _ _ := []

theorem honestService_accepts (Ω : Type) [Fintype Ω] (lam : ℕ) (h : Fintype.card Ω = 2 ^ lam) :
    (∀ σ, (honestService Ω lam h).verify () σ true = true) ∧ ∃ σ, (honestService Ω lam h).verify () σ false = false :=
  ⟨fun _ => rfl, ⟨Classical.arbitrary _, rfl⟩⟩

theorem prob_false {Ω : Type*} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  simp [prob]

theorem prob_eq_one_of_forall {Ω : Type*} [Fintype Ω] [Nonempty Ω] {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  unfold prob
  rw [Finset.filter_true_of_mem fun ω _ => h ω, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

/-- A one-slot program whose slot is the product `f`. -/
def oneProd {V : Type} (f : (V → ZMod (2 ^ 32)) → ZMod (2 ^ 32)) : SlpG V (ZMod (2 ^ 32)) 1 :=
  ⟨fun _ => true, fun _ => f, fun _ _ => 0, fun _ => ⟨0, fun _ => 0⟩⟩

theorem oneProd_val {V : Type} [Fintype V] (f : (V → ZMod (2 ^ 32)) → ZMod (2 ^ 32)) (z : V → ZMod (2 ^ 32))
    (j : Fin 1) : (oneProd f).val z j = f z := by
  rw [SlpG.val]
  simp [oneProd]

theorem dP_leftOnly {n L np B n' : ℕ} (pc : Fin n → Fin B → Fin L → ZMod (2 ^ 32)) (w : Fin np → ZMod (2 ^ 32))
    (i : Fin n) (t : Fin B) (j : Fin L) : LeftOnly fun z => (dP (n' := n') pc w i t j).eval z := by
  intro z z' h
  simp only [BitAff.eval, dP, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, zero_mul, Finset.sum_const_zero, h]

theorem dX_rightOnly {n L np B n' : ℕ} (vq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (t : Fin B) (j : Fin L)
    (l : Fin n') : RightOnly fun z => (dX (n := n) (np := np) vq t j l).eval z := by
  intro z z' h
  simp only [BitAff.eval, dX, Fintype.sum_sum_type, Sum.elim_inl, Sum.elim_inr, zero_mul, Finset.sum_const_zero, h]

/-- The one-cell design's target: its cell-block's prefix word at zero constants. -/
def dTarget : (DBits 1 1 7 1 1 → ZMod (2 ^ 32)) → ZMod (2 ^ 32) := dPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) wRe 0 0 0

theorem dTarget_eq (z : DBits 1 1 7 1 1 → ZMod (2 ^ 32)) :
    dTarget z = (dP (n' := 1) (fun _ _ _ => 0) wRe 0 0 0).eval z * (dX (n := 1) (np := 7) (fun _ _ _ => 0) 0 0 0).eval z := by
  simp [dTarget, dPrefix, dIncr]

theorem dTarget_oneSided : OneSided dTarget :=
  ⟨_, _, dP_leftOnly _ wRe 0 0 0, dX_rightOnly _ 0 0 0, dTarget_eq⟩

theorem dTarget_affProd : AffProd dTarget := ⟨_, _, dTarget_eq⟩

theorem dCells_eq (c c' : DCells 1 1 1) : c = c' := Subsingleton.elim _ _

theorem dPrefix_cell (c : DCells 1 1 1) :
    dPrefix (fun _ _ _ => (0 : ZMod (2 ^ 32))) (fun _ _ _ => 0) wRe c.1 c.2.1 c.2.2.val = dTarget := by
  obtain rfl := dCells_eq c (0, 0, 0)
  rfl

/-- The honest staged run: one round of one stage, its one product slot the target, so its error tables vanish. -/
def dStagedRun : StagedRun 1 1 1 1 7 1 1 (ZMod (2 ^ 32)) where
  play := .round (.stage (fun c z => (oneProd dTarget).val z 0 -
    dPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) wRe c.1 c.2.1 c.2.2.val z) fun _ => .done) fun _ => .done
  Bind _ _ := False
  ZK _ _ := False
  V _ := 1
  N _ _ _ := 1
  prog _ _ _ := oneProd dTarget
  pc _ _ _ := fun _ _ _ => 0
  vq _ _ _ := fun _ _ _ => 0
  word _ _ _ _ := 0

theorem dStagedRun_accepts (x : Paths (DStage 1 1 1 7 1 1) 1) : dStagedRun.play.game.accepts x := by
  obtain ⟨⟨⟨U, ⟨⟩⟩, s, c⟩, ⟨⟩⟩ := x
  obtain rfl := Subsingleton.elim s 0
  refine ⟨?_, trivial⟩
  show (oneProd dTarget).val _ 0 - dPrefix _ _ wRe c.1 c.2.1 c.2.2.val _ = 0
  rw [oneProd_val, dPrefix_cell, sub_self]

theorem dStagedRun_errAt (x : Paths (DStage 1 1 1 7 1 1) 1) (w : Fin 1) (s : Fin 1) (c : DCells 1 1 1) :
    dStagedRun.play.errAt x w s c = fun z => (oneProd dTarget).val z 0 -
      dPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) wRe c.1 c.2.1 c.2.2.val z := by
  obtain rfl := Subsingleton.elim w 0
  obtain rfl := Subsingleton.elim s 0
  rfl

theorem card_dStage : Fintype.card (Paths (DStage 1 1 1 7 1 1) 1) = 2 ^ 8 := by
  rw [Fintype.card_eq_nat_card]
  simp [Nat.card_eq_fintype_card, Fintype.card_finset]

theorem mergeAt_one {ι : Fin 1 → Type} [∀ t, Fintype (ι t)] [∀ t, DecidableEq (ι t)] (U : ∀ s, Finset (ι s))
    (z : ι 0 → ZMod (2 ^ 32)) : (fun v => mergeAt U 0 z ⟨0, v⟩) = z := by
  funext v
  simp [mergeAt]

theorem oCells_eq (c c' : OCells 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)) : c = c' := by
  obtain ⟨t, d⟩ := c
  obtain ⟨t', d'⟩ := c'
  obtain rfl : t = t' := Subsingleton.elim _ _
  obtain rfl : d = d' := dCells_eq d d'
  rfl

/-- The one-cell overlapped design's target: the one draw's `dTarget`. -/
def oTarget : (OBits 1 (fun _ => 1) 1 7 (fun _ => 1) (fun _ => 1) → ZMod (2 ^ 32)) → ZMod (2 ^ 32) :=
  fun z => dTarget fun v => z ⟨0, v⟩

theorem oTarget_merge (U : ODraws 1 (fun _ => 1) 1 7 (fun _ => 1) (fun _ => 1)) :
    (fun z => oTarget (mergeAt U 0 z)) = dTarget :=
  funext fun z => congrArg dTarget (mergeAt_one U z)

theorem oPrefix_cell (c : OCells 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)) :
    (fun z : OBits 1 (fun _ => 1) 1 7 (fun _ => 1) (fun _ => 1) → ZMod (2 ^ 32) =>
      dPrefix (fun _ _ _ => (0 : ZMod (2 ^ 32))) (fun _ _ _ => 0) wRe c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) =
      oTarget := by
  obtain rfl := oCells_eq c ⟨0, 0, 0, 0⟩
  rfl

/-- The honest overlapped run: one round of one draw, its one product slot the target. -/
def oRun : OverlapRun 1 1 (fun _ => 1) 1 7 (fun _ => 1) (fun _ => 1) (ZMod (2 ^ 32)) where
  play := .round (fun c z => (oneProd oTarget).val z 0 -
    dPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) wRe c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) fun _ => .done
  Bind _ _ := False
  ZK _ _ := False
  V _ := 1
  N _ _ := 1
  prog _ _ := oneProd oTarget
  pc _ _ _ := fun _ _ _ => 0
  vq _ _ _ := fun _ _ _ => 0
  word _ _ _ := 0

theorem oRun_accepts (x : Paths (OSpot 1 (fun _ => 1) 1 7 (fun _ => 1) fun _ => 1) 1) : oRun.play.game.accepts x := by
  refine ⟨?_, trivial⟩
  show (oneProd oTarget).val _ 0 - _ = 0
  rw [oneProd_val, sub_eq_zero]
  exact (congrFun (oPrefix_cell x.1.1) _).symm

theorem oRun_errAt (x : Paths (OSpot 1 (fun _ => 1) 1 7 (fun _ => 1) fun _ => 1) 1) (w : Fin 1)
    (c : OCells 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)) :
    oRun.play.errAt x w c = fun z => (oneProd oTarget).val z 0 -
      dPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) wRe c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩ := by
  obtain rfl := Subsingleton.elim w 0
  rfl

theorem card_oSpot : Fintype.card (Paths (OSpot 1 (fun _ => 1) 1 7 (fun _ => 1) fun _ => 1) 1) = 2 ^ 8 := by
  rw [Fintype.card_eq_nat_card]
  simp [Nat.card_eq_fintype_card, Fintype.card_finset, Fintype.card_sigma, Fintype.card_pi]

theorem jX_leftOnly {n L B n' : ℕ} (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32)) (i : Fin n) (t : Fin B) (j : Fin L) :
    LeftOnly fun z => jX (n' := n') aq i t j z := by
  intro z z' h
  simp only [jX, h]

theorem jW_rightOnly {n L B n' : ℕ} (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (t : Fin B) (j : Fin L)
    (l : Fin n') : RightOnly fun z => jW (n := n) wq t j l z := by
  intro z z' h
  simp only [jW, h]

/-- The one-cell call's target: its cell-block's prefix word at zero operands. -/
def jTarget : (JBits 1 1 1 1 → ZMod (2 ^ 32)) → ZMod (2 ^ 32) := jPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) 0 0 0

theorem jTarget_eq (z : JBits 1 1 1 1 → ZMod (2 ^ 32)) :
    jTarget z = jX (n' := 1) (fun _ _ _ => 0) 0 0 0 z * jW (n := 1) (fun _ _ _ => 0) 0 0 0 z := by
  simp [jTarget, jPrefix, jIncr]

theorem jTarget_oneSided : OneSided jTarget :=
  ⟨_, _, jX_leftOnly _ 0 0 0, jW_rightOnly _ 0 0 0, jTarget_eq⟩

theorem jPrefix_cell (c : DCells 1 1 1) :
    jPrefix (fun _ _ _ => (0 : ZMod (2 ^ 32))) (fun _ _ _ => 0) c.1 c.2.1 c.2.2.val = jTarget := by
  obtain rfl := dCells_eq c (0, 0, 0)
  rfl

/-- The honest call run: one round of one call, its one product slot the target. -/
def jStagedRun : JStagedRun 1 1 1 1 1 1 (ZMod (2 ^ 32)) where
  play := .round (.stage (fun c z => (oneProd jTarget).val z 0 -
    jPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) c.1 c.2.1 c.2.2.val z) fun _ => .done) fun _ => .done
  Bind _ _ := False
  ZK _ _ := False
  V _ := 1
  N _ _ _ := 1
  prog _ _ _ := oneProd jTarget
  aq _ _ _ := fun _ _ _ => 0
  wq _ _ _ := fun _ _ _ => 0
  word _ _ _ _ := 0

theorem jStagedRun_accepts (x : Paths (JStage 1 1 1 1 1) 1) : jStagedRun.play.game.accepts x := by
  obtain ⟨⟨⟨U, ⟨⟩⟩, s, c⟩, ⟨⟩⟩ := x
  obtain rfl := Subsingleton.elim s 0
  refine ⟨?_, trivial⟩
  show (oneProd jTarget).val _ 0 - jPrefix _ _ c.1 c.2.1 c.2.2.val _ = 0
  rw [oneProd_val, jPrefix_cell, sub_self]

theorem jStagedRun_errAt (x : Paths (JStage 1 1 1 1 1) 1) (w : Fin 1) (s : Fin 1) (c : DCells 1 1 1) :
    jStagedRun.play.errAt x w s c = fun z => (oneProd jTarget).val z 0 -
      jPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) c.1 c.2.1 c.2.2.val z := by
  obtain rfl := Subsingleton.elim w 0
  obtain rfl := Subsingleton.elim s 0
  rfl

theorem card_jStage : Fintype.card (Paths (JStage 1 1 1 1 1) 1) = 2 ^ 10 := by
  rw [Fintype.card_eq_nat_card]
  simp [Nat.card_eq_fintype_card, Fintype.card_finset]

/-- The one-cell overlapped call's target: the one call's `jTarget`. -/
def ojTarget : (OJBits 1 (fun _ => 1) 1 (fun _ => 1) (fun _ => 1) → ZMod (2 ^ 32)) → ZMod (2 ^ 32) :=
  fun z => jTarget fun v => z ⟨0, v⟩

theorem ojTarget_merge (U : OJDraws 1 (fun _ => 1) 1 (fun _ => 1) (fun _ => 1)) :
    (fun z => ojTarget (mergeAt U 0 z)) = jTarget :=
  funext fun z => congrArg jTarget (mergeAt_one U z)

theorem ojPrefix_cell (c : OCells 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)) :
    (fun z : OJBits 1 (fun _ => 1) 1 (fun _ => 1) (fun _ => 1) → ZMod (2 ^ 32) =>
      jPrefix (fun _ _ _ => (0 : ZMod (2 ^ 32))) (fun _ _ _ => 0) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) =
      ojTarget := by
  obtain rfl := oCells_eq c ⟨0, 0, 0, 0⟩
  rfl

/-- The honest overlapped call run: one round of one call draw, its one product slot the target. -/
def ojRun : JOverlapRun 1 1 (fun _ => 1) 1 (fun _ => 1) (fun _ => 1) (ZMod (2 ^ 32)) where
  play := .round (fun c z => (oneProd ojTarget).val z 0 -
    jPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) fun _ => .done
  Bind _ _ := False
  ZK _ _ := False
  V _ := 1
  N _ _ := 1
  prog _ _ := oneProd ojTarget
  aq _ _ _ := fun _ _ _ => 0
  wq _ _ _ := fun _ _ _ => 0
  word _ _ _ := 0

theorem ojRun_accepts (x : Paths (OJSpot 1 (fun _ => 1) 1 (fun _ => 1) fun _ => 1) 1) :
    ojRun.play.game.accepts x := by
  refine ⟨?_, trivial⟩
  show (oneProd ojTarget).val _ 0 - _ = 0
  rw [oneProd_val, sub_eq_zero]
  exact (congrFun (ojPrefix_cell x.1.1) _).symm

theorem ojRun_errAt (x : Paths (OJSpot 1 (fun _ => 1) 1 (fun _ => 1) fun _ => 1) 1) (w : Fin 1)
    (c : OCells 1 (fun _ => 1) (fun _ => 1) (fun _ => 1)) :
    ojRun.play.errAt x w c = fun z => (oneProd ojTarget).val z 0 -
      jPrefix (fun _ _ _ => 0) (fun _ _ _ => 0) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩ := by
  obtain rfl := Subsingleton.elim w 0
  rfl

theorem card_ojSpot : Fintype.card (Paths (OJSpot 1 (fun _ => 1) 1 (fun _ => 1) fun _ => 1) 1) = 2 ^ 10 := by
  rw [Fintype.card_eq_nat_card]
  simp [Nat.card_eq_fintype_card, Fintype.card_finset, Fintype.card_sigma, Fintype.card_pi]

end Honest

theorem StagedFloorPerProverHonest : Pouw.Guarantees.RelativeSecurity.StagedFloorPerProverHonest := by
  intro sha kz
  let S := honestService (Paths (DStage 1 1 1 7 1 1) 1) 8 card_dStage
  refine ⟨S, Equiv.refl _, fun _ _ _ _ => dStagedRun, (), true, false, ?_, ?_, ?_, ?_, honestService_accepts _ _ _⟩
  · intro Ω' _ M _ r
    refine ⟨fun w => ?_, fun w => ?_, fun x _ => Or.inl (dStagedRun_accepts x)⟩ <;>
      simp only [dStagedRun, prob_false, zero_mul, add_zero, le_refl]
  · intro Ω' _ M _ r x
    refine ⟨fun w s j _ => dTarget_affProd, fun w s c => dStagedRun_errAt x w s c, ?_⟩
    intro w s c c' _ _ hne
    exact absurd (Subsingleton.elim _ _) hne
  · intro Ω' _ M _ r x
    refine ⟨fun w s j _ => dTarget_oneSided, fun w s c => dStagedRun_errAt x w s c, ?_⟩
    intro w s c c' i hne
    exact absurd (dCells_eq c c') hne
  · intro Ω' _ M _ r x
    refine ⟨le_rfl, ?_⟩
    simp [dStagedRun, oneProd, Machine.charge, funMachine]
    exact Finset.card_le_one.mpr fun a _ b _ => Subsingleton.elim a b

theorem OverlapFloorPerProverHonest : Pouw.Guarantees.RelativeSecurity.OverlapFloorPerProverHonest := by
  intro sha kz
  let S := honestService (Paths (OSpot 1 (fun _ => 1) 1 7 (fun _ => 1) fun _ => 1) 1) 8 card_oSpot
  refine ⟨S, Equiv.refl _, fun _ _ _ _ => oRun, (), true, false, ?_, ?_, ?_, ?_, honestService_accepts _ _ _⟩
  · intro Ω' _ M _ r
    refine ⟨fun w => ?_, fun w => ?_, fun x _ => Or.inl (oRun_accepts x)⟩ <;>
      simp only [oRun, prob_false, zero_mul, add_zero, le_refl]
  · intro Ω' _ M _ r x
    refine ⟨fun w j _ U t => ?_, fun w c => oRun_errAt x w c, ?_⟩
    · obtain rfl := Subsingleton.elim t 0
      obtain ⟨A, B, hAB⟩ := dTarget_affProd
      exact ⟨A, B, fun z => (congrFun (oTarget_merge U) z).trans (hAB z)⟩
    · intro w W c c' _ _ hne
      exact (hne (by rw [oCells_eq c c'])).elim
  · intro Ω' _ M _ r x
    refine ⟨fun w j _ U t => ?_, fun w c => oRun_errAt x w c, ?_⟩
    · obtain rfl := Subsingleton.elim t 0
      show OneSided fun z => oTarget (mergeAt U 0 z)
      rw [oTarget_merge]
      exact dTarget_oneSided
    · intro w W c c' i hne
      exact absurd (oCells_eq c c') hne
  · intro Ω' _ M _ r x
    refine ⟨le_rfl, ?_⟩
    simp [oRun, oneProd, Machine.charge, funMachine]
    exact Finset.card_le_one.mpr fun a _ b _ => Subsingleton.elim a b

theorem StagedFloorJ10RPerProverHonest : Pouw.Guarantees.RelativeSecurity.StagedFloorJ10RPerProverHonest := by
  intro sha kz
  let S := honestService (Paths (JStage 1 1 1 1 1) 1) 10 card_jStage
  refine ⟨S, Equiv.refl _, fun _ _ _ _ => jStagedRun, (), true, false, ?_, ?_, ?_, honestService_accepts _ _ _⟩
  · intro Ω' _ M _ r
    refine ⟨fun w => ?_, fun w => ?_, fun x _ => Or.inl (jStagedRun_accepts x)⟩ <;>
      simp only [jStagedRun, prob_false, zero_mul, add_zero, le_refl]
  · intro Ω' _ M _ r x
    refine ⟨fun w s j _ => jTarget_oneSided, fun w s c => jStagedRun_errAt x w s c, ?_⟩
    intro w s c c' i hne
    exact absurd (dCells_eq c c') hne
  · intro Ω' _ M _ r x
    refine ⟨le_rfl, ?_⟩
    simp [jStagedRun, oneProd, Machine.charge, funMachine]
    exact Finset.card_le_one.mpr fun a _ b _ => Subsingleton.elim a b

theorem OverlapFloorJ10RPerProverHonest : Pouw.Guarantees.RelativeSecurity.OverlapFloorJ10RPerProverHonest := by
  intro sha kz
  let S := honestService (Paths (OJSpot 1 (fun _ => 1) 1 (fun _ => 1) fun _ => 1) 1) 10 card_ojSpot
  refine ⟨S, Equiv.refl _, fun _ _ _ _ => ojRun, (), true, false, ?_, ?_, ?_, honestService_accepts _ _ _⟩
  · intro Ω' _ M _ r
    refine ⟨fun w => ?_, fun w => ?_, fun x _ => Or.inl (ojRun_accepts x)⟩ <;>
      simp only [ojRun, prob_false, zero_mul, add_zero, le_refl]
  · intro Ω' _ M _ r x
    refine ⟨fun w j _ U t => ?_, fun w c => ojRun_errAt x w c, ?_⟩
    · obtain rfl := Subsingleton.elim t 0
      show OneSided fun z => ojTarget (mergeAt U 0 z)
      rw [ojTarget_merge]
      exact jTarget_oneSided
    · intro w W c c' i hne
      exact absurd (oCells_eq c c') hne
  · intro Ω' _ M _ r x
    refine ⟨le_rfl, ?_⟩
    simp [ojRun, oneProd, Machine.charge, funMachine]
    exact Finset.card_le_one.mpr fun a _ b _ => Subsingleton.elim a b

theorem FloorPerProverNontrivial : Pouw.Guarantees.RelativeSecurity.FloorPerProverNontrivial := by
  intro S _ _ Ω R e sha kz F hacc hF h
  obtain ⟨c₀⟩ := ‹Nonempty S.Com›
  obtain ⟨a₀⟩ := ‹Nonempty S.Ans›
  have h1 := h Unit (fun _ => (c₀, a₀)) fun _ _ _ => rfl
  rw [prob_eq_one_of_forall] at h1
  swap
  · exact fun ω => ⟨hacc _ _ _, by simpa [Machine.charge, funMachine] using hF⟩
  rw [ENNReal.ofReal_one] at h1
  unfold breakPr at h1
  simp only [Finset.univ_unique, Finset.sum_singleton, Fintype.card_unit, Nat.cast_one, inv_one, one_mul,
    add_zero, mul_zero, zero_mul] at h1
  have h2 : (1 : ENNReal) ≤ ENNReal.ofReal ((2 : ℝ) ^ 40)⁻¹ := h1.trans (min_le_right _ _)
  rw [ENNReal.one_le_ofReal] at h2
  norm_num at h2

end Pouw.SecurityProofs.RelativeSecurity
