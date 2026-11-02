import Mathlib
import Definitions.Pouw.RelativeSecurityJ10R
import Definitions.Pouw.Guarantees.RelativeSecurity
import Proofs.Pouw.RelativeSecurity
import Proofs.Pouw.RelativeSecurityFork

/-!
# Relative security at call scope under J10-R: the floors of record, proved on the call design

Theorem K's one-sided chain (`roundsHyp_of_stagedP`, `roundsHyp_of_overlapP`) reads the design in four places: the
frame is injective (`dFrame_injective`), a one-sided product splits there (`OneSided.splitProd`), the target is
`⟨x, y⟩` plus junk there (`split_dPrefix`), and the charges add up (`SlpG.card_full_le`). Each is restated here on
J10-R's call design (`jFrame_injective`, `OneSided.splitProdJ`, `split_jPrefix`, `SlpG.card_full_leJ`), and the rest of
the chain, the product-deficit games and their luck 0.854, reads the frame only as a parameter.

`split_jPrefix` is where the call design differs. Segment `t`'s increment at cell `(i, l)` is
`Σ_j s8 (A + x_j) · s8 (W + y_j + U_j)`, `x` row `i`'s dither bits and `y` column `l`'s coin bits of segment `t`; the
int8 wrap moves a value by a multiple of 256, so mod 2 it is `Σ_j (A + x_j)(W + U_j + y_j)` (`map_jS8`), whose
two-sided part is `⟨x, y⟩`. `U` reads neither side of any frame, and the earlier segments' increments read no bit of
segment `t`'s frame, so no plane weight enters: any ring map onto `F₂` does.

The floors per prover take the same chain one adversary and one value of its coins at a time
(`roundsSlice_of_stagedPJ`, `roundsSlice_of_overlapPJ`) into `serviceFloorPerProver_breakPr_of_slices`, at the
adversary's fork finders.
-/

namespace Pouw.RelativeSecurity

open Finset

section J10RDesign

variable {n L B n' : ℕ}

/-- The int8 wrap is invisible mod 2: it subtracts a multiple of 256. -/
theorem map_jS8 (φ : ZMod (2 ^ 32) →+* ZMod 2) (x : ZMod (2 ^ 32)) : φ (jS8 x) = φ x := by
  have h1 : φ x = ((x.val : ℕ) : ZMod 2) := by
    conv_lhs => rw [← ZMod.natCast_zmod_val x]
    exact map_natCast φ _
  have h2 : (((x.val + 128) % 256 : ℕ) : ZMod 2) = ((x.val : ℕ) : ZMod 2) := by
    rw [ZMod.natCast_eq_natCast_iff', Nat.mod_mod_of_dvd _ (by norm_num : 2 ∣ 256)]
    omega
  have h128 : φ (128 : ZMod (2 ^ 32)) = 0 := by
    rw [map_ofNat φ 128]
    decide
  rw [jS8, map_sub, map_natCast, h128, sub_zero, h2, h1]

theorem jFrame_injective (c : DCells n n' B) : Function.Injective (jFrame (L := L) c) := by
  rintro (j | j) (j' | j') h <;> simp_all [jFrame]

/-- A one-sided product of any form is split at every cell-block's frame: its factor on the rows' bits is blind to
the columns' coins, and its factor on the columns' bits to the rows' dither bits. -/
theorem OneSided.splitProdJ {Ar : Type} [CommRing Ar] (φ : Ar →+* ZMod 2) (c : DCells n n' B)
    {π : (JBits n L B n' → Ar) → Ar} (hπ : OneSided π) : SplitProd (jFrame c) φ π := by
  obtain ⟨g, h, hg, hh, hgh⟩ := hπ
  refine ⟨fun S => φ (g (bitInd S)), fun S => φ (h (bitInd S)), fun S S' hS => ?_, fun S S' hS => ?_,
    fun S => by rw [hgh, map_mul]⟩
  · exact congrArg φ (hg (bitInd S) (bitInd S') fun a => by
      unfold bitInd
      rw [if_congr (hS (.inl a) fun j => by simp [jFrame]) rfl rfl])
  · exact congrArg φ (hh (bitInd S) (bitInd S') fun b => by
      unfold bitInd
      rw [if_congr (hS (.inr b) fun j => by simp [jFrame]) rfl rfl])

/-- A function of one of `U`'s bytes reads no bit of any frame: it is blind to both sides. -/
theorem jFrame_blind_u (c : DCells n n' B) (g : (Fin 8 → ZMod (2 ^ 32)) → ZMod 2) (l : Fin n') (j : Fin L) :
    XBlind (jFrame c) (fun S => g fun p => bitInd S (.inr (.inr (l, j, p)) : JBits n L B n')) ∧
      YBlind (jFrame c) (fun S => g fun p => bitInd S (.inr (.inr (l, j, p)) : JBits n L B n')) := by
  refine ⟨fun S S' hS => congrArg g (funext fun p => ?_), fun S S' hS => congrArg g (funext fun p => ?_)⟩
  · unfold bitInd
    rw [if_congr (hS _ fun j' => by simp [jFrame]) rfl rfl]
  · unfold bitInd
    rw [if_congr (hS _ fun j' => by simp [jFrame]) rfl rfl]

/-- **The call's target is `⟨x, y⟩` plus junk at its cell-block's frame**, mod 2 through any `φ`: segment `t`'s
increment is `Σ_j (A_j + x_j)(Q_j + y_j)` mod 2, `Q_j` the registered weight and `U`'s value, and the earlier
segments' increments read no bit of the frame. -/
theorem split_jPrefix (φ : ZMod (2 ^ 32) →+* ZMod 2) (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32))
    (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (c : DCells n n' B) :
    Split (jFrame c) fun S => φ (jPrefix aq wq c.1 c.2.1 c.2.2.val (bitInd S)) -
      (fun j => frameBits (jFrame c) S (.inl j)) ⬝ᵥ fun j => frameBits (jFrame c) S (.inr j) := by
  classical
  obtain ⟨i, l, t⟩ := c
  set fr := jFrame (L := L) (i, l, t) with hfr
  let xb : Fin B → Fin L → Finset (JBits n L B n') → ZMod 2 := fun t' j S =>
    if (.inl (i, t', j) : JBits n L B n') ∈ S then 1 else 0
  let yb : Fin B → Fin L → Finset (JBits n L B n') → ZMod 2 := fun t' j S =>
    if (.inr (.inl (t', j, l)) : JBits n L B n') ∈ S then 1 else 0
  let Q : Fin B → Fin L → Finset (JBits n L B n') → ZMod 2 := fun t' j S =>
    φ (wq t' j l) + φ (jU fun p => bitInd S (.inr (.inr (l, j, p)) : JBits n L B n'))
  have hX : ∀ t' j S, φ (jX aq i t' j (bitInd S)) = φ (aq i t' j) + xb t' j S := fun t' j S => by
    simp only [jX, map_jS8, map_add, map_bitInd, xb]
  have hW : ∀ t' j S, φ (jW wq t' j l (bitInd S)) = Q t' j S + yb t' j S := fun t' j S => by
    simp only [jW, map_jS8, map_add, map_bitInd, Q, yb]
    ring
  have hQy : ∀ t' j, YBlind fr (Q t' j) := fun t' j =>
    (YBlind.const fr _).add (jFrame_blind_u (i, l, t) (fun u => φ (jU u)) l j).2
  have hxy : ∀ t' j, YBlind fr (xb t' j) := fun t' j => YBlind.bit fun j' h => by simp [hfr, jFrame] at h
  have hyx : ∀ t' j, XBlind fr (yb t' j) := fun t' j => XBlind.bit fun j' h => by simp [hfr, jFrame] at h
  have hyy : ∀ t' j, t' ≠ t → YBlind fr (yb t' j) := fun t' j ht => YBlind.bit fun j' h => by
    simp only [hfr, jFrame, Sum.elim_inr, Sum.inr.injEq, Sum.inl.injEq, Prod.mk.injEq] at h
    exact ht h.1.symm
  have hpiece : ∀ t' j, Split fr fun S =>
      (if t'.val ≤ t.val then (φ (aq i t' j) + xb t' j S) * (Q t' j S + yb t' j S) else 0) -
        if t' = t then xb t j S * yb t j S else 0 := fun t' j => by
    by_cases ht : t' = t
    · subst ht
      refine ((Split.of_y (((YBlind.const fr (φ (aq i t' j))).add (hxy t' j)).mul (hQy t' j))).add
        (Split.of_x ((XBlind.const fr (φ (aq i t' j))).mul (hyx t' j)))).congr fun S => ?_
      simp only [le_refl, ↓reduceIte]
      ring
    · by_cases hle : t'.val ≤ t.val
      · refine (Split.of_y (((YBlind.const fr (φ (aq i t' j))).add (hxy t' j)).mul
          ((hQy t' j).add (hyy t' j ht)))).congr fun S => ?_
        simp only [hle, ht, ↓reduceIte, sub_zero]
      · exact (Split.of_y (YBlind.const fr 0)).congr fun S => by simp [hle, ht]
  refine (Split.sum univ fun t' _ => Split.sum univ fun j _ => hpiece t' j).congr fun S => ?_
  simp only [sum_sub_distrib]
  have hdot : ((fun j => frameBits fr S (.inl j)) ⬝ᵥ fun j => frameBits fr S (.inr j)) =
      ∑ t' : Fin B, ∑ j, if t' = t then xb t j S * yb t j S else 0 := by
    rw [sum_comm]
    simp only [sum_ite_eq', mem_univ, ↓reduceIte, dotProduct]
    rfl
  rw [hdot]
  congr 1
  simp only [jPrefix, jIncr, map_sum, apply_ite φ, map_zero, map_mul]
  refine sum_congr rfl fun t' _ => ?_
  split_ifs with h
  · exact sum_congr rfl fun j _ => by rw [hX, hW]
  · simp

/-- **A one-sided call's error has a parity form at every cell-block's frame**, its terms the products charged to the
cell-block's word (`charged`): any straight-line program over `Z/2³²` on the call's bits, its products one-sided of
any form, through any `φ` onto `F₂`. -/
theorem SlpG.parityForm_errJ (φ : ZMod (2 ^ 32) →+* ZMod 2) {N : ℕ} (P : SlpG (JBits n L B n') (ZMod (2 ^ 32)) N)
    (hOk : ∀ j, P.isProd j = true → OneSided (P.prod j)) (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32))
    (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (j : Fin N) (c : DCells n n' B) :
    ParityForm (jFrame c) φ (charged (jFrame c) φ (P.coef j) P.prod).card
      fun z => P.val z j - jPrefix aq wq c.1 c.2.1 c.2.2.val z := by
  have h := parityForm_sum (jFrame_injective c) φ (P.coef j) P.prod (P.aff j)
    (fun i hi => OneSided.splitProdJ φ c (hOk i (P.isProd_of_coef_ne_zero j i hi))) _ (split_jPrefix φ aq wq c)
  have e : (fun z => P.val z j - jPrefix aq wq c.1 c.2.1 c.2.2.val z) =
      fun z => ∑ i, P.coef j i * P.prod i z + (P.aff j).eval z - jPrefix aq wq c.1 c.2.1 c.2.2.val z :=
    funext fun z => by rw [P.val_eq]
  rw [e]
  exact h

open Classical in
/-- **Charge-pure one-sided calls add up**: the call's program has at least `L` products per cell-block not short of
products. -/
theorem SlpG.card_full_leJ (φ : ZMod (2 ^ 32) →+* ZMod 2) {N : ℕ} (P : SlpG (JBits n L B n') (ZMod (2 ^ 32)) N)
    (hOk : ∀ j, P.isProd j = true → OneSided (P.prod j)) (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32))
    (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (word : DCells n n' B → Fin N) (hpure : P.ChargePureJ φ word) :
    L * (univ.filter fun c => ¬ParShort (jFrame c) φ
      fun z => P.val z (word c) - jPrefix aq wq c.1 c.2.1 c.2.2.val z).card ≤
      (univ.filter fun j => P.isProd j = true).card := by
  refine _root_.Pouw.RelativeSecurity.card_full_le (fun c => charged (jFrame c) φ (P.coef (word c)) P.prod) _
    (fun c i hi => mem_filter.mpr ⟨mem_univ _, P.isProd_of_coef_ne_zero _ _ (ne_zero_of_map (mem_filter.mp hi).2.1)⟩)
    (fun c c' hne => disjoint_left.mpr fun i hi hi' =>
      hpure c c' i hne (mem_filter.mp hi).2.1 (mem_filter.mp hi).2.2 (mem_filter.mp hi').2.1
        (mem_filter.mp hi').2.2) _ fun c hc => ?_
  by_contra hlt
  exact hc ⟨_, not_le.mp hlt, P.parityForm_errJ φ hOk aq wq (word c) c⟩

end J10RDesign

section J10ROverlapDesign

variable {St : ℕ} {n B n' : Fin St → ℕ} {L : ℕ}

/-- **An overlapped one-sided run's error has a parity form at every cell-block's frame of every call**, at every bit
point `W` of the other calls: its terms the products charged to the word there. -/
theorem SlpG.parityForm_err_ovJ (φ : ZMod (2 ^ 32) →+* ZMod 2) {N : ℕ}
    (P : SlpG (OJBits St n L B n') (ZMod (2 ^ 32)) N) (hOk : ∀ j, P.isProd j = true → OneSidedAtJ (P.prod j))
    (W : OJDraws St n L B n') (aq : ∀ t, Fin (n t) → Fin (B t) → Fin L → ZMod (2 ^ 32))
    (wq : ∀ t, Fin (B t) → Fin L → Fin (n' t) → ZMod (2 ^ 32)) (j : Fin N) (c : OCells St n n' B) :
    ParityForm (jFrame c.2) φ (charged (jFrame c.2) φ (P.coef j) fun i z => P.prod i (mergeAt W c.1 z)).card
      fun z => P.val (mergeAt W c.1 z) j - jPrefix (aq c.1) (wq c.1) c.2.1 c.2.2.1 c.2.2.2.val z := by
  have h := parityForm_sum (jFrame_injective c.2) φ (P.coef j) (fun i z => P.prod i (mergeAt W c.1 z))
    ((P.aff j).restrictAt W c.1)
    (fun i hi => OneSided.splitProdJ φ c.2 (hOk i (P.isProd_of_coef_ne_zero j i hi) W c.1)) _
    (split_jPrefix φ (aq c.1) (wq c.1) c.2)
  have e : (fun z => P.val (mergeAt W c.1 z) j - jPrefix (aq c.1) (wq c.1) c.2.1 c.2.2.1 c.2.2.2.val z) =
      fun z => ∑ i, P.coef j i * P.prod i (mergeAt W c.1 z) + ((P.aff j).restrictAt W c.1).eval z -
        jPrefix (aq c.1) (wq c.1) c.2.1 c.2.2.1 c.2.2.2.val z :=
    funext fun z => by rw [P.val_eq, BitAff.eval_mergeAt]
  rw [e]
  exact h

open Classical in
/-- **Charge-pure one-sided overlapped calls add up**: at every bit point of the draws, the round's run has at least
`L` products per cell-block, of any call, not short of products. -/
theorem SlpG.card_full_le_ovJ (φ : ZMod (2 ^ 32) →+* ZMod 2) {N : ℕ}
    (P : SlpG (OJBits St n L B n') (ZMod (2 ^ 32)) N) (hOk : ∀ j, P.isProd j = true → OneSidedAtJ (P.prod j))
    (W : OJDraws St n L B n') (aq : ∀ t, Fin (n t) → Fin (B t) → Fin L → ZMod (2 ^ 32))
    (wq : ∀ t, Fin (B t) → Fin L → Fin (n' t) → ZMod (2 ^ 32)) (word : OCells St n n' B → Fin N)
    (hpure : P.ChargePureAtJ φ word) :
    L * (univ.filter fun c : OCells St n n' B => ¬ParShort (jFrame c.2) φ
      fun z => P.val (mergeAt W c.1 z) (word c) - jPrefix (aq c.1) (wq c.1) c.2.1 c.2.2.1 c.2.2.2.val z).card ≤
      (univ.filter fun j => P.isProd j = true).card := by
  refine _root_.Pouw.RelativeSecurity.card_full_le
    (fun c => charged (jFrame c.2) φ (P.coef (word c)) fun i z => P.prod i (mergeAt W c.1 z)) _
    (fun c i hi => mem_filter.mpr ⟨mem_univ _, P.isProd_of_coef_ne_zero _ _ (ne_zero_of_map (mem_filter.mp hi).2.1)⟩)
    (fun c c' hne => disjoint_left.mpr fun i hi hi' =>
      hpure W c c' i hne (mem_filter.mp hi).2.1 (mem_filter.mp hi).2.2 (mem_filter.mp hi').2.1
        (mem_filter.mp hi').2.2) _ fun c hc => ?_
  by_contra hlt
  exact hc ⟨_, not_le.mp hlt, P.parityForm_err_ovJ φ hOk W aq wq (word c) c⟩

end J10ROverlapDesign

/-- **The window's staged calls, one-sided operands of any form, under J10-R**, a named hypothesis: `RoundsHypKStP` on
the call design, each stage a call (`JStage`), its target `jPrefix` and charging purity at `jFrame`. -/
def RoundsHypJStP (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n L B n' : ℕ)
    (φ : ZMod (2 ^ 32) →+* ZMod 2) : Prop :=
  ∃ e : S.Seed ≃ Paths (JStage St n L B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : StagedPlay (DCells n n' B) (JBits n L B n') (ZMod (2 ^ 32)) St R)
      (Bind ZK : Fin R → Paths (JStage St n L B n') R → Prop) (V : Paths (JStage St n L B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (JBits n L B n') (ZMod (2 ^ 32)) (N w s))
        (aq : Fin R → Fin St → Fin n → Fin B → Fin L → ZMod (2 ^ 32))
        (wq : Fin R → Fin St → Fin B → Fin L → Fin n' → ZMod (2 ^ 32))
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → OneSided ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - jPrefix (aq w s) (wq w s) c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s, (prog w s).ChargePureJ φ (word w s)) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- `roundsHyp_of_stagedPJ` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypJStP`'s cost line at that slice. -/
theorem roundsSlice_of_stagedPJ {R St : ℕ} {Vmax κ : ℝ} {n L B n' : ℕ} {φ : ZMod (2 ^ 32) →+* ZMod 2}
    (P : StagedPlay (DCells n n' B) (JBits n L B n') (ZMod (2 ^ 32)) St R) (acc : Paths (JStage St n L B n') R → Prop)
    (cost : Paths (JStage St n L B n') R → ℝ) (Bind ZK : Fin R → Paths (JStage St n L B n') R → Prop)
    (V : Paths (JStage St n L B n') R → ℝ) (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x)
    (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → Fin St → ℕ) (prog : ∀ w s, SlpG (JBits n L B n') (ZMod (2 ^ 32)) (N w s))
        (aq : Fin R → Fin St → Fin n → Fin B → Fin L → ZMod (2 ^ 32))
        (wq : Fin R → Fin St → Fin B → Fin L → Fin n' → ZMod (2 ^ 32))
        (word : ∀ w s, DCells n n' B → Fin (N w s)),
        (∀ w s j, (prog w s).isProd j = true → OneSided ((prog w s).prod j)) ∧
        (∀ w s c, P.errAt x w s c =
          fun z => (prog w s).val z (word w s c) - jPrefix (aq w s) (wq w s) c.1 c.2.1 c.2.2.val z) ∧
        (∀ w s, (prog w s).ChargePureJ φ (word w s)) ∧
        κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hSt : 0 < St) (hκ : 0 ≤ κ) :
    RoundsSlice (427 / 500) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax acc cost Bind ZK := by
  classical
  refine ⟨P.gameP jFrame φ, V, (P.gameP jFrame φ).expBound_of_staged (by norm_num)
    (P.game_stagedP (fun c => jFrame_injective c) φ),
    fun x hx => (hacc x hx).imp_left (P.accepts_gameP jFrame φ x).mpr, hV, fun x => ?_⟩
  obtain ⟨N, prog, aq, wq, word, hOk, hE, hpure, hcharge⟩ := hK x
  set C : ℝ := ((n * n' * B : ℕ) : ℝ)
  have hC : C = (Fintype.card (DCells n n' B) : ℝ) := by
    simp [C, Fintype.card_prod, mul_assoc]
  have hprod : ∀ w s, (L : ℝ) * (C * (1 - parDef jFrame φ (P.errAt x w s))) ≤
      ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) := fun w s => by
    have h1 := (prog w s).card_full_leJ φ (hOk w s) (aq w s) (wq w s) (word w s) (hpure w s)
    have h2 := card_not_ge fun c => ParShort (jFrame c) φ (P.errAt x w s c)
    rw [hC, parDef]
    refine (mul_le_mul_of_nonneg_left h2 (Nat.cast_nonneg L)).trans ?_
    simp only [hE w s] at h1 ⊢
    exact_mod_cast h1
  have hSt' : (St : ℝ) ≠ 0 := by exact_mod_cast hSt.ne'
  have hdef : (P.gameP jFrame φ).deficiency x =
      (St : ℝ)⁻¹ * ∑ w, ∑ s, parDef jFrame φ (P.errAt x w s) := by
    rw [RoundGame.deficiency_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl fun w _ => ?_
    rw [P.dAt_gameP jFrame φ x w, Finset.mul_sum]
  have hsum : ((R : ℝ) - (P.gameP jFrame φ).deficiency x) * (St * (κ * L * C)) =
      κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - parDef jFrame φ (P.errAt x w s))) := by
    have e2 : ∑ w : Fin R, ∑ s : Fin St, (L : ℝ) * (C * (1 - parDef jFrame φ (P.errAt x w s))) =
        L * C * (R * St - ∑ w, ∑ s, parDef jFrame φ (P.errAt x w s)) := by
      simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, ← Finset.mul_sum]
      ring
    rw [hdef, e2]
    field_simp
  calc ((R : ℝ) - (P.gameP jFrame φ).deficiency x) * (St * (κ * L * C)) - V x - R * 0
      = κ * ∑ w, ∑ s, (L : ℝ) * (C * (1 - parDef jFrame φ (P.errAt x w s))) - V x := by rw [hsum]; ring
    _ ≤ κ * ∑ w, ∑ s, ((univ.filter fun j => (prog w s).isProd j = true).card : ℝ) - V x := by
      gcongr with w _ s
      exact hprod w s
    _ ≤ cost x := hcharge

/-- **Staged one-sided calls give the window's cost line** under J10-R: `roundsHyp_of_stagedP` on the call design,
`RoundsHyp` at `p = 0.854` with `F₀ = St κ L n n' B` and no slack (`roundsSlice_of_stagedPJ`, slice by slice). -/
theorem roundsHyp_of_stagedPJ {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n L B n' : ℕ}
    {φ : ZMod (2 ^ 32) →+* ZMod 2} (h : RoundsHypJStP Mc S R St Vmax δb δz κ n L B n' φ) (hSt : 0 < St)
    (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (JStage St n L B n') R (427 / 500) (St * (κ * L * ((n * n' * B : ℕ) : ℝ))) 0 Vmax δb δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_stagedPJ P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hSt hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

/-- **The window's overlapped calls, one-sided operands of any form, under J10-R**, a named hypothesis:
`RoundsHypKOvP` on the call design (`OJSpot`), its targets `jPrefix` and charging purity across calls at `jFrame`. -/
def RoundsHypJOvP (Mc : Machine) (S : Service) (R St : ℕ) (Vmax δb δz κ : ℝ) (n : Fin St → ℕ) (L : ℕ)
    (B n' : Fin St → ℕ) (φ : ZMod (2 ^ 32) →+* ZMod 2) : Prop :=
  ∃ e : S.Seed ≃ Paths (OJSpot St n L B n') R, ∀ (Ω' : Type) [Fintype Ω']
    (M : Mc.Alg (S.Seed × Ω') (S.Com × S.Ans)), (∀ σ σ' r, (Mc.eval M (σ, r)).1 = (Mc.eval M (σ', r)).1) →
    ∀ r : Ω', ∃ (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => JBits (n t) L (B t) (n' t))
        (ZMod (2 ^ 32)) R)
      (Bind ZK : Fin R → Paths (OJSpot St n L B n') R → Prop) (V : Paths (OJSpot St n L B n') R → ℝ),
      (∀ w, prob (Bind w) ≤ δb) ∧ (∀ w, prob (ZK w) ≤ δz) ∧
      (∀ x, S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true →
        P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) ∧
      (∀ x, V x ≤ Vmax) ∧
      ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OJBits St n L B n') (ZMod (2 ^ 32)) (N w))
        (aq : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → ZMod (2 ^ 32))
        (wq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → ZMod (2 ^ 32))
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → OneSidedAtJ ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          jPrefix (aq w c.1) (wq w c.1) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w, (prog w).ChargePureAtJ φ (word w)) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ Mc.charge M (e.symm x, r)

/-- `roundsHyp_of_overlapPJ` on one slice (`RoundsSlice`): one adversary at one value of its coins, its acceptance
`acc`, charge `cost` and breaks named, and `RoundsHypJOvP`'s cost line at that slice. -/
theorem roundsSlice_of_overlapPJ {R St : ℕ} {Vmax κ : ℝ} {n B n' : Fin St → ℕ} {L : ℕ}
    {φ : ZMod (2 ^ 32) →+* ZMod 2}
    (P : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => JBits (n t) L (B t) (n' t)) (ZMod (2 ^ 32)) R)
    (acc : Paths (OJSpot St n L B n') R → Prop) (cost : Paths (OJSpot St n L B n') R → ℝ)
    (Bind ZK : Fin R → Paths (OJSpot St n L B n') R → Prop) (V : Paths (OJSpot St n L B n') R → ℝ)
    (hacc : ∀ x, acc x → P.game.accepts x ∨ ∃ w, Bind w x ∨ ZK w x) (hV : ∀ x, V x ≤ Vmax)
    (hK : ∀ x, ∃ (N : Fin R → ℕ) (prog : ∀ w, SlpG (OJBits St n L B n') (ZMod (2 ^ 32)) (N w))
        (aq : Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → ZMod (2 ^ 32))
        (wq : Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → ZMod (2 ^ 32))
        (word : ∀ w, OCells St n n' B → Fin (N w)),
        (∀ w j, (prog w).isProd j = true → OneSidedAtJ ((prog w).prod j)) ∧
        (∀ w c, P.errAt x w c = fun z => (prog w).val z (word w c) -
          jPrefix (aq w c.1) (wq w c.1) c.2.1 c.2.2.1 c.2.2.2.val fun v => z ⟨c.1, v⟩) ∧
        (∀ w, (prog w).ChargePureAtJ φ (word w)) ∧
        κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x ≤ cost x)
    (hκ : 0 ≤ κ) :
    RoundsSlice (427 / 500) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax acc cost Bind ZK := by
  classical
  let fr : (c : OCells St n n' B) → Fin L ⊕ Fin L → JBits (n c.1) L (B c.1) (n' c.1) := fun c => jFrame c.2
  refine ⟨P.gameP fr φ, V, RoundGame.expBound_of_pathBound _
    ((P.gameP fr φ).pathBound_of_passBound (P.passBound_gameP (fun c => jFrame_injective c.2) φ)),
    fun x hx => (hacc x hx).imp_left (P.accepts_gameP fr φ x).mpr, hV, fun x => ?_⟩
  obtain ⟨N, prog, aq, wq, word, hOk, hE, hpure, hcharge⟩ := hK x
  set C : ℝ := (Fintype.card (OCells St n n' B) : ℝ)
  let full : Fin R → OJDraws St n L B n' → OCells St n n' B → Prop := fun w W c =>
    ¬ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z)
  have hprod : ∀ w W, (L : ℝ) * ((univ.filter fun c => full w W c).card : ℝ) ≤
      ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) := fun w W => by
    have h1 := (prog w).card_full_le_ovJ φ (hOk w) W (aq w) (wq w) (word w) (hpure w)
    simp only [full, fr, hE, mergeAt_mk]
    exact_mod_cast h1
  have hcor : ∀ w, C * (1 - ovDefP fr φ (P.errAt x w)) ≤
      expect fun W : OJDraws St n L B n' => ((univ.filter fun c => full w W c).card : ℝ) := fun w => by
    have hd : ovDefP fr φ (P.errAt x w) = expect fun W : OJDraws St n L B n' =>
        prob fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z) := by
      rw [ovDefP, prob_eq_expect, expect_prod_swap]
      congr 1
      funext W
      rw [prob_eq_expect]
    have ha : C * (1 - ovDefP fr φ (P.errAt x w)) = expect fun W : OJDraws St n L B n' =>
        C + -C * prob fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z) := by
      rw [hd, expect_const_add_mul]
      ring
    rw [ha]
    refine expect_mono fun W => ?_
    have := card_not_ge fun c : OCells St n n' B => ParShort (fr c) φ fun z => P.errAt x w c (mergeAt W c.1 z)
    simp only [full]
    linarith
  have hdef : (P.gameP fr φ).deficiency x = ∑ w, ovDefP fr φ (P.errAt x w) := by
    rw [RoundGame.deficiency_eq_sum]
    exact Finset.sum_congr rfl fun w _ => P.dAt_gameP fr φ x w
  calc ((R : ℝ) - (P.gameP fr φ).deficiency x) * (κ * L * C) - V x - R * 0
      = κ * ∑ w, (L : ℝ) * (C * (1 - ovDefP fr φ (P.errAt x w))) - V x := by
        rw [hdef]
        simp only [mul_sub, mul_one, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul, ← Finset.mul_sum]
        ring
    _ ≤ κ * ∑ w, ((univ.filter fun j => (prog w).isProd j = true).card : ℝ) - V x := by
      gcongr with w
      refine (mul_le_mul_of_nonneg_left (hcor w) (Nat.cast_nonneg L)).trans ?_
      rw [← expect_mul_left]
      exact expect_le_of_le (Nat.cast_nonneg _) fun W => hprod w W
    _ ≤ cost x := hcharge

/-- **Overlapping one-sided calls give the window's cost line** under J10-R: `roundsHyp_of_overlapP` on the call
design, `RoundsHyp` at `p = 0.854` with `F₀ = κ L C`, `C` the round's cell-blocks over all its calls
(`roundsSlice_of_overlapPJ`, slice by slice). -/
theorem roundsHyp_of_overlapPJ {Mc : Machine} {S : Service} {R St : ℕ} {Vmax δb δz κ : ℝ} {n B n' : Fin St → ℕ}
    {L : ℕ} {φ : ZMod (2 ^ 32) →+* ZMod 2} (h : RoundsHypJOvP Mc S R St Vmax δb δz κ n L B n' φ) (hκ : 0 ≤ κ) :
    RoundsHyp Mc S (OJSpot St n L B n') R (427 / 500) (κ * L * (Fintype.card (OCells St n n' B) : ℝ)) 0 Vmax δb
      δz := by
  obtain ⟨e, h⟩ := h
  refine ⟨e, fun Ω' _ M hcom r => ?_⟩
  obtain ⟨P, Bind, ZK, V, hB, hZ, hacc, hV, hK⟩ := h Ω' M hcom r
  obtain ⟨G, V', hG, hacc', hV', hcost⟩ := roundsSlice_of_overlapPJ P
    (fun x => S.verify (Mc.eval M (e.symm x, r)).1 (e.symm x) (Mc.eval M (e.symm x, r)).2 = true)
    (fun x => Mc.charge M (e.symm x, r)) Bind ZK V hacc hV hK hκ
  exact ⟨G, Bind, ZK, V', hG, hB, hZ, hacc', hV', hcost⟩

end Pouw.RelativeSecurity

namespace Pouw.SecurityProofs.RelativeSecurity

open Pouw.RelativeSecurity Pouw.Assumptions

theorem StagedFloorKOneSidedJ10R : Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedJ10R := by
  intro Mc S R St n L B n' e run δb δz κ Vmax hSt hκ hC hδb hδz hS hK hP
  have hKSt : RoundsHypJStP Mc S R St Vmax δb δz κ n L B n' toF2 := ⟨e, fun Ω' _ M hcom r => by
    obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
    exact ⟨(run Ω' M r).play, (run Ω' M r).Bind, (run Ω' M r).ZK, (run Ω' M r).V, hB, hZ, hacc,
      fun x => (hP Ω' M hcom r x).1, fun x =>
        ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).aq x, (run Ω' M r).wq x, (run Ω' M r).word x,
          (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩⟩⟩
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]; exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have h := serviceFloor_of_rounds (roundsHyp_of_stagedPJ hKSt hSt hκ.le) (by norm_num) hF₀ hδb hδz
    (274 * Real.log 2)
  rw [mul_zero, sub_zero] at h
  exact ModelB.ServiceFloor.mono h (by linarith [ModelB.exp_parity_274])

theorem OverlapFloorKOneSidedJ10R : Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedJ10R := by
  intro Mc S R St L n B n' e run δb δz κ Vmax hκ hC hδb hδz hS hK hP
  have hKOv : RoundsHypJOvP Mc S R St Vmax δb δz κ n L B n' toF2 := ⟨e, fun Ω' _ M hcom r => by
    obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
    exact ⟨(run Ω' M r).play, (run Ω' M r).Bind, (run Ω' M r).ZK, (run Ω' M r).V, hB, hZ, hacc,
      fun x => (hP Ω' M hcom r x).1, fun x =>
        ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).aq x, (run Ω' M r).wq x, (run Ω' M r).word x,
          (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩⟩⟩
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]; exact mul_pos hκ (by exact_mod_cast hC)
  have h := serviceFloor_of_rounds (roundsHyp_of_overlapPJ hKOv hκ.le) (by norm_num) hF₀ hδb hδz
    (274 * Real.log 2)
  rw [mul_zero, sub_zero, card_oCells] at h
  exact ModelB.ServiceFloor.mono h (by linarith [ModelB.exp_parity_274])

theorem StagedFloorKOneSidedJ10RPerProver :
    Pouw.Guarantees.RelativeSecurity.StagedFloorKOneSidedJ10RPerProver := by
  intro Mc S R St n L B n' e run sha kz δb βb δz βz κ Vmax hSt hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < (St : ℝ) * (κ * L * ((n * n' * B : ℕ) : ℝ)) := by
    rw [mul_assoc]; exact mul_pos (by exact_mod_cast hSt) (mul_pos hκ (by exact_mod_cast hC))
  have h := serviceFloorPerProver_breakPr_of_slices (p := 427 / 500) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz) (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_stagedPJ (φ := toF2) (run Ω' M r).play
        _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).aq x, (run Ω' M r).wq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        hSt hκ.le⟩)
    (by norm_num) hF₀ (274 * Real.log 2) ModelB.exp_parity_274
  rwa [mul_zero, sub_zero] at h

theorem OverlapFloorKOneSidedJ10RPerProver :
    Pouw.Guarantees.RelativeSecurity.OverlapFloorKOneSidedJ10RPerProver := by
  intro Mc S R St L n B n' e run sha kz δb βb δz βz κ Vmax hκ hC _ _ _ _ hS hK hP
  have hF₀ : 0 < κ * L * (Fintype.card (OCells St n n' B) : ℝ) := by
    rw [mul_assoc, card_oCells]; exact mul_pos hκ (by exact_mod_cast hC)
  have h := serviceFloorPerProver_breakPr_of_slices (p := 427 / 500) (sl := 0) (Vmax := Vmax) (δb := δb) (δz := δz) e sha
    (forkFinders S e sha 1) (forkFinders S e sha kz) (fun Ω' _ M hcom r => by
      obtain ⟨hB, hZ, hacc⟩ := hS Ω' M hcom r
      exact ⟨(run Ω' M r).Bind, (run Ω' M r).ZK, hB, hZ, roundsSlice_of_overlapPJ (φ := toF2) (run Ω' M r).play
        _ _ _ _ (run Ω' M r).V hacc (fun x => (hP Ω' M hcom r x).1) (fun x =>
          ⟨(run Ω' M r).N x, (run Ω' M r).prog x, (run Ω' M r).aq x, (run Ω' M r).wq x, (run Ω' M r).word x,
            (hK Ω' M hcom r x).1, (hK Ω' M hcom r x).2.1, (hK Ω' M hcom r x).2.2, (hP Ω' M hcom r x).2⟩)
        hκ.le⟩)
    (by norm_num) hF₀ (274 * Real.log 2) ModelB.exp_parity_274
  rwa [mul_zero, sub_zero, card_oCells] at h

end Pouw.SecurityProofs.RelativeSecurity
