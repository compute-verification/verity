import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic.GCongr
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# Refutation of the forward-only (`q_inv = 0`) P3 lemma

One forward-only adversary against the two-layer chain on `Fin 8` (`wc = 64`, `m = 512`, `s = 1`, `D = 3`,
`Q = 4`, `k = 2`, `W = 0`) answers two challenges with probability at least `2^-448` (`pr_attack`), while every
statement below bounds it by `8·56⁴·2^-512 ≈ 2^-485.8` (`bound_lt`). Refuted: the pinned
`Pous.Guarantees.InvExPostFactoTwoLayer`, the `q_inv = 0` case of `pebbling/InvertibleExPostFactoDraft.lean`'s
`inv_ex_post_facto_bounded_inv` (`ForwardOnlyBound`), and that file's placeholder `ForwardBound`.
-/

namespace PousP3Forward

open scoped ENNReal

theorem two_pow_mul_inv_pow (a b : ℕ) (h : a ≤ b) :
    (2 : ℝ≥0∞) ^ a * (2⁻¹ : ℝ≥0∞) ^ b = (2⁻¹ : ℝ≥0∞) ^ (b - a) := by
  obtain ⟨c, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [pow_add, ← mul_assoc, ← mul_pow, ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_pow, one_mul,
    Nat.add_sub_cancel_left]

end PousP3Forward

namespace PousP3Forward

open Pous Pous.Invertible Pous.Labelling
open scoped ENNReal

/-! ### The graph: a chain on `Fin 8` -/

/-- The chain on `Fin 8`: the parent of `b` is `b − 1`. -/
def chain8 : TopoDAG 8 where
  par b := Finset.univ.filter fun a => a.val + 1 = b.val
  par_lt b a ha := by
    rw [Finset.mem_filter] at ha
    exact Fin.lt_def.2 (by omega)

theorem mem_par_chain8 (a b : Fin 8) : a ∈ chain8.par b ↔ a.val + 1 = b.val := by
  simp [chain8]

/-! ### White/green hardness of the two-layer chain: `WhiteGreenHard chain8 1 3 2` -/

section Hard

variable {C : ℕ → Finset (Fin 2 × Fin 8)}

/-- A top node has no green child, so it is placed only forward, which needs every lower node. -/
theorem top_step (hC : DRG.IsPebbling node2 (drg2 chain8) green2 C) (r : ℕ) (hlow : ∃ a, (0, a) ∉ C r) (b : Fin 8)
    (hb : (1, b) ∈ C (r + 1)) : (1, b) ∈ C r := by
  by_contra hnot
  obtain ⟨a, ha⟩ := hlow
  rcases hC r (1, b) hb hnot with hfwd | ⟨⟨u, hu⟩, -⟩
  · exact ha (hfwd (0, a) (Or.inl ⟨rfl, rfl⟩))
  · simp [green2] at hu

/-- A lower node is placed forward only after its white parent; its reverse move needs every top node. So each
round adds at most node 0 and one successor per pebbled lower node. -/
theorem low_step (hC : DRG.IsPebbling node2 (drg2 chain8) green2 C) (r : ℕ) (htop : ∃ b, (1, b) ∉ C r) :
    (Finset.univ.filter fun a : Fin 8 => ((0, a) : Fin 2 × Fin 8) ∈ C (r + 1)).card ≤
      2 * (Finset.univ.filter fun a : Fin 8 => ((0, a) : Fin 2 × Fin 8) ∈ C r).card + 1 := by
  classical
  set L := Finset.univ.filter fun a : Fin 8 => ((0, a) : Fin 2 × Fin 8) ∈ C r
  have hsub : (Finset.univ.filter fun a : Fin 8 => ((0, a) : Fin 2 × Fin 8) ∈ C (r + 1)) ⊆
      L ∪ ({0} ∪ L.image fun a => a + 1) := by
    intro a ha
    rw [Finset.mem_filter] at ha
    by_cases hold : ((0, a) : Fin 2 × Fin 8) ∈ C r
    · exact Finset.mem_union_left _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hold⟩)
    · refine Finset.mem_union_right _ ?_
      rcases hC r (0, a) ha.2 hold with hfwd | ⟨-, hrev⟩
      · by_cases ha0 : a = 0
        · exact Finset.mem_union_left _ (Finset.mem_singleton.2 ha0)
        · refine Finset.mem_union_right _ (Finset.mem_image.2 ⟨⟨a.val - 1, by omega⟩, ?_, ?_⟩)
          · refine Finset.mem_filter.2 ⟨Finset.mem_univ _, hfwd _ (Or.inr ⟨0, ⟨a.val - 1, by omega⟩, a, ?_, rfl, rfl⟩)⟩
            rw [drg2, mem_par_chain8]
            have : a.val ≠ 0 := fun h => ha0 (Fin.ext h)
            simp only
            omega
          · have : a.val ≠ 0 := fun h => ha0 (Fin.ext h)
            ext
            simp only [Fin.val_add, Fin.val_one]
            omega
      · obtain ⟨b, hb⟩ := htop
        exact absurd (hrev (1, b) ⟨rfl, rfl⟩).1 hb
  calc _ ≤ (L ∪ ({0} ∪ L.image fun a => a + 1)).card := Finset.card_le_card hsub
    _ ≤ L.card + (1 + L.card) := by
        refine (Finset.card_union_le _ _).trans (Nat.add_le_add_left ?_ _)
        exact (Finset.card_union_le _ _).trans (Nat.add_le_add (by simp) Finset.card_image_le)
    _ = 2 * L.card + 1 := by ring

theorem exists_not_mem_of_card_lt {S : Finset (Fin 8)} (h : S.card < 8) : ∃ a, a ∉ S := by
  by_contra hno
  push Not at hno
  have : S = Finset.univ := Finset.eq_univ_of_forall hno
  rw [this, Finset.card_univ, Fintype.card_fin] at h
  omega

/-- **Hardness.** From at most one pebble, the lower layer holds at most `1`, `3`, `7` pebbles at rounds 0, 1, 2, so
no top pebble is placed by round 3. -/
theorem hard : WhiteGreenHard chain8 1 3 2 := by
  classical
  intro C hC h0
  let L : ℕ → Finset (Fin 8) := fun r => Finset.univ.filter fun a => ((0, a) : Fin 2 × Fin 8) ∈ C r
  let T : ℕ → Finset (Fin 8) := fun r => Finset.univ.filter fun b => ((1, b) : Fin 2 × Fin 8) ∈ C r
  have hL0 : (L 0).card ≤ 1 := le_trans (Finset.card_le_card_of_injOn (fun a => ((0, a) : Fin 2 × Fin 8))
    (fun a ha => (Finset.mem_filter.1 ha).2) fun a _ b _ h => by simpa using h) h0
  have hT0 : (T 0).card ≤ 1 := le_trans (Finset.card_le_card_of_injOn (fun b => ((1, b) : Fin 2 × Fin 8))
    (fun b hb => (Finset.mem_filter.1 hb).2) fun a _ b _ h => by simpa using h) h0
  have hTsub : ∀ r, (L r).card < 8 → T (r + 1) ⊆ T r := fun r hr b hb =>
    Finset.mem_filter.2 ⟨Finset.mem_univ _, top_step hC r (exists_not_mem_of_card_lt hr |>.imp fun a ha h =>
      ha (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩)) b (Finset.mem_filter.1 hb).2⟩
  have hLstep : ∀ r, (T r).card < 8 → (L (r + 1)).card ≤ 2 * (L r).card + 1 := fun r hr =>
    low_step hC r (exists_not_mem_of_card_lt hr |>.imp fun b hb h => hb (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
  have hT1 : T 1 ⊆ T 0 := hTsub 0 (by omega)
  have hL1 : (L 1).card ≤ 3 := by
    have h : (L 1).card ≤ 2 * (L 0).card + 1 := hLstep 0 (by omega)
    omega
  have hT2 : T 2 ⊆ T 1 := hTsub 1 (by omega)
  have hL2 : (L 2).card ≤ 7 := by
    have h : (L 2).card ≤ 2 * (L 1).card + 1 := hLstep 1 (lt_of_le_of_lt (Finset.card_le_card hT1) (by omega))
    omega
  have hT3 : T 3 ⊆ T 2 := hTsub 2 (by omega)
  have : (T 3).card ≤ 1 := (Finset.card_le_card (hT3.trans (hT2.trans hT1))).trans hT0
  exact lt_of_le_of_lt this (by norm_num)

end Hard

end PousP3Forward

namespace PousP3Forward

open Pous Pous.Invertible Pous.Labelling
open scoped ENNReal

/-! ### Inverse-query counting (verbatim from `pebbling/InvertibleExPostFactoDraft.lean`) -/

/-- Whether a P3 query is an inverse (`P⁻¹`) query. -/
def isInv {n wc : ℕ} (q : (p3Model n wc).Qry) : Bool :=
  match q with
  | .inr (.inr _) => true
  | _ => false

/-- The number of `P⁻¹` queries a program makes on the path taken when queries are answered by `f`. -/
def invWork {n wc : ℕ} {α : Type} (f : (p3Model n wc).Qry → (p3Model n wc).Ans) : OracleProg (p3Model n wc) α → ℕ
  | .ret _ => 0
  | .round qs k => (qs.filter fun q => isInv q).length + invWork f (k (qs.map f))

/-- A program makes at most `q_inv` inverse queries, against every primitive. -/
def InvBounded {n wc : ℕ} {α : Type} (q_inv : ℕ) (p : OracleProg (p3Model n wc) α) : Prop :=
  ∀ ω : (p3Model n wc).Ω, invWork ((p3Model n wc).answer ω) p ≤ q_inv

/-! ### The adversary -/

def z64 : Bits 64 := fun _ => false
def zLab : Lab 8 64 := fun _ => z64
def W0 : Fin 8 → Lab 8 64 := fun _ => zLab

/-- The online program for challenge 1, from the stored top label `σ = c²_0`: it computes `c¹_0` and guesses
`c¹_u[1] = 0` for `u ≥ 1`. Three rounds, four forward queries. -/
noncomputable def prog1 (σ : Lab 8 64) : OracleProg (p3Model 8 64) (Lab 8 64) :=
  .round [Sum.inl (0, 0, fun _ => none), Sum.inl (1, 1, fun a => if a ∈ chain8.par 1 then some σ else none)]
    fun as =>
      .round [Sum.inr (Sum.inl (labXor (W0 0) (as.getD 0 zLab)))] fun bs =>
        .round [Sum.inr (Sum.inl (labXor (fun u => if u = 0 then (bs.getD 0 zLab) 1 else z64) (as.getD 1 zLab)))]
          fun cs => .ret (cs.getD 0 zLab)

/-- The preprocessor stores the top label `c²_0` (512 bits). -/
noncomputable def A₁ (ω : (p3Model 8 64).Ω) : Bits 512 := codeEquiv 8 64 (topLabel chain8 W0 ω 0)

/-- The online programs: challenge 0 from the state, challenge 1 by `prog1`, the others give up. -/
noncomputable def A₂ (σ : Bits 512) (v : Fin 8) : OracleProg (p3Model 8 64) (Lab 8 64) :=
  if v = 0 then .ret ((codeEquiv 8 64).symm σ) else if v = 1 then prog1 ((codeEquiv 8 64).symm σ) else .ret zLab

theorem A₂_bounded (σ : Bits 512) (v : Fin 8) : (A₂ σ v).Bounded 3 4 := by
  intro ω
  unfold A₂
  split_ifs
  · exact ⟨Nat.zero_le _, Nat.zero_le _⟩
  · exact ⟨le_of_eq rfl, le_of_eq rfl⟩
  · exact ⟨Nat.zero_le _, Nat.zero_le _⟩

theorem A₂_invBounded (σ : Bits 512) (v : Fin 8) : InvBounded 0 (A₂ σ v) := by
  intro ω
  unfold A₂
  split_ifs
  · exact le_of_eq rfl
  · exact le_of_eq rfl
  · exact le_of_eq rfl

/-! ### The event: the guess is right whenever `c¹_u[1] = 0` for `u = 1..7` -/

theorem c1_eq (H : KQ 8 64 → Lab 8 64) (P : Equiv.Perm (Lab 8 64)) (u : Fin 8) :
    c1 chain8 H P W0 u =
      P (labXor (W0 u) (H (0, u, fun a => if a ∈ chain8.par u then some (c1 chain8 H P W0 a) else none))) := by
  rw [c1]
  simp only [dite_eq_ite]

theorem c2_eq (H : KQ 8 64 → Lab 8 64) (P : Equiv.Perm (Lab 8 64)) (v : Fin 8) :
    c2 chain8 H P W0 v =
      P (labXor (transpose (c1 chain8 H P W0) v)
        (H (1, v, fun a => if a ∈ chain8.par v then some (c2 chain8 H P W0 a) else none))) := by
  rw [c2]
  simp only [dite_eq_ite]

theorem correct_of (H : KQ 8 64 → Lab 8 64) (P : Equiv.Perm (Lab 8 64))
    (hE : ∀ u : Fin 8, 1 ≤ u.val → u.val ≤ 7 → c1 chain8 H P W0 u 1 = z64) :
    2 ≤ (Finset.univ.filter fun v =>
      (A₂ (A₁ (H, P)) v).run ((p3Model 8 64).answer (H, P)) = topLabel chain8 W0 (H, P) v).card := by
  have h0 : (A₂ (A₁ (H, P)) 0).run ((p3Model 8 64).answer (H, P)) = topLabel chain8 W0 (H, P) 0 := by
    simp [A₂, A₁, OracleProg.run]
  have h1 : (A₂ (A₁ (H, P)) 1).run ((p3Model 8 64).answer (H, P)) = topLabel chain8 W0 (H, P) 1 := by
    have hq1 : (fun a : Fin 8 => if a ∈ chain8.par 1 then some ((codeEquiv 8 64).symm (A₁ (H, P))) else none) =
        fun a => if a ∈ chain8.par 1 then some (c2 chain8 H P W0 a) else none := by
      funext a
      split_ifs with ha
      · rw [mem_par_chain8] at ha
        have : a = 0 := Fin.ext (by simp at ha ⊢; omega)
        subst this
        simp [A₁, topLabel]
      · rfl
    have hq0 : (fun a : Fin 8 => if a ∈ chain8.par 0 then some (c1 chain8 H P W0 a) else none) = fun _ => none := by
      funext a
      simp [mem_par_chain8]
    have hT : (fun u : Fin 8 => if u = 0 then (P (labXor (W0 0) (H (0, 0, fun _ => none)))) 1 else z64) =
        transpose (c1 chain8 H P W0) 1 := by
      funext u
      simp only [transpose]
      split_ifs with hu
      · subst hu
        rw [c1_eq, hq0]
      · exact (hE u (by have : u.val ≠ 0 := fun h => hu (Fin.ext h); omega) (by omega)).symm
    have hne : (1 : Fin 8) ≠ 0 := by decide
    simp only [A₂, hne, ↓reduceIte, prog1, OracleProg.run]
    rw [topLabel, c2_eq, ← hT, ← hq1]
    rfl
  refine le_trans (le_of_eq ?_) (Finset.card_le_card (s := {0, 1}) fun v hv => ?_)
  · decide
  · rw [Finset.mem_insert, Finset.mem_singleton] at hv
    rw [Finset.mem_filter]
    rcases hv with rfl | rfl
    · exact ⟨Finset.mem_univ _, h0⟩
    · exact ⟨Finset.mem_univ _, h1⟩

end PousP3Forward

namespace PousP3Forward

open Pous Pous.Invertible Pous.Labelling
open scoped ENNReal

/-- One decidable equality on the oracle's domain, so the oracle space has a single `Fintype` instance. -/
noncomputable instance decEqKQ : DecidableEq (KQ 8 64) := Classical.decEq _

/-! ### Each guessed component costs exactly `2^-64` -/

section Count

variable (P : Equiv.Perm (Lab 8 64))

/-- The real query of lower node `u`. -/
def rq (H : KQ 8 64 → Lab 8 64) (u : Fin 8) : KQ 8 64 :=
  (0, u, fun a => if a ∈ chain8.par u then some (c1 chain8 H P W0 a) else none)

theorem labXor_cancel (a b : Lab 8 64) : labXor a (labXor a b) = b := by
  funext v j
  simp only [labXor, bxor]
  cases a v j <;> cases b v j <;> rfl

/-- A lower label depends only on the oracle at the real queries of the node and its ancestors. -/
theorem c1_congr {H H' : KQ 8 64 → Lab 8 64} (u : Fin 8) (h : ∀ u' ≤ u, H' (rq P H u') = H (rq P H u')) :
    c1 chain8 H' P W0 u = c1 chain8 H P W0 u := by
  induction u using WellFoundedLT.induction with
  | _ u ih =>
    have hq : rq P H' u = rq P H u := by
      unfold rq
      congr 1
      congr 1
      funext a
      split_ifs with ha
      · rw [ih a (chain8.par_lt u a ha) fun u' hu' => h u' (hu'.trans (chain8.par_lt u a ha).le)]
      · rfl
    rw [c1_eq, c1_eq]
    change P (labXor (W0 u) (H' (rq P H' u))) = P (labXor (W0 u) (H (rq P H u)))
    rw [hq, h u le_rfl]

theorem rq_congr {H H' : KQ 8 64 → Lab 8 64} (u : Fin 8) (h : ∀ u' < u, H' (rq P H u') = H (rq P H u')) :
    rq P H' u = rq P H u := by
  unfold rq
  congr 1
  congr 1
  funext a
  split_ifs with ha
  · rw [c1_congr P a fun u' hu' => h u' (lt_of_le_of_lt hu' (chain8.par_lt u a ha))]
  · rfl

/-- The first `j` guesses are right: `c¹_u[1] = 0` for `1 ≤ u ≤ j`. -/
def Ev (j : ℕ) (H : KQ 8 64 → Lab 8 64) : Prop := ∀ u : Fin 8, 1 ≤ u.val → u.val ≤ j → c1 chain8 H P W0 u 1 = z64

noncomputable instance instDecidablePredForallKQOfNatNatLabEv (j : ℕ) : DecidablePred (Ev P j) := Classical.decPred _

set_option linter.constructorNameAsVariable false in
/-- **One more guess costs `2^64`.** Resample the oracle at node `j + 1`'s real query so that its label's
component 1 is 0, and remember the old component: an injection. -/
theorem card_Ev_step (j : ℕ) (hj : j < 7) :
    (Finset.univ.filter (Ev P j)).card ≤ (Finset.univ.filter (Ev P (j + 1))).card * Fintype.card (Bits 64) := by
  set u₁ : Fin 8 := ⟨j + 1, by omega⟩ with hu₁
  let x' : (KQ 8 64 → Lab 8 64) → Lab 8 64 := fun H => Function.update (c1 chain8 H P W0 u₁) 1 z64
  let upd : (KQ 8 64 → Lab 8 64) → (KQ 8 64 → Lab 8 64) := fun H =>
    Function.update H (rq P H u₁) (labXor (W0 u₁) (P.symm (x' H)))
  -- the update keeps the earlier real queries, and node `j + 1`'s real query
  have hagree : ∀ H, ∀ u' < u₁, upd H (rq P H u') = H (rq P H u') := by
    intro H u' hu'
    apply Function.update_of_ne
    intro h
    have := congrArg (fun q : KQ 8 64 => q.2.1) h
    simp only [rq] at this
    exact absurd this (ne_of_lt hu')
  have hrq : ∀ H, rq P (upd H) u₁ = rq P H u₁ := fun H => rq_congr P u₁ (hagree H)
  have hlow : ∀ H, ∀ u : Fin 8, u.val ≤ j → c1 chain8 (upd H) P W0 u = c1 chain8 H P W0 u := by
    intro H u hu
    exact c1_congr P u fun u' hu' => hagree H u' (lt_of_le_of_lt hu' (by rw [Fin.lt_def]; simp [hu₁]; omega))
  have hnew : ∀ H, c1 chain8 (upd H) P W0 u₁ = x' H := by
    intro H
    rw [c1_eq]
    change P (labXor (W0 u₁) (upd H (rq P (upd H) u₁))) = x' H
    rw [hrq]
    simp only [upd, Function.update_self]
    rw [labXor_cancel, Equiv.apply_symm_apply]
  have hcard : ((Finset.univ.filter (Ev P (j + 1))) ×ˢ (Finset.univ : Finset (Bits 64))).card =
      (Finset.univ.filter (Ev P (j + 1))).card * Fintype.card (Bits 64) := by
    rw [Finset.card_product, Finset.card_univ]
  rw [← hcard]
  refine Finset.card_le_card_of_injOn (fun H => (upd H, c1 chain8 H P W0 u₁ 1)) ?_ ?_
  · intro H hH
    rw [Finset.mem_coe, Finset.mem_filter] at hH
    rw [Finset.mem_coe, Finset.mem_product, Finset.mem_filter]
    simp only [Finset.mem_univ, true_and, and_true]
    intro u hu1 huj
    by_cases hle : u.val ≤ j
    · rw [hlow H u hle]
      exact hH.2 u hu1 hle
    · have : u = u₁ := Fin.ext (by simp [hu₁]; omega)
      subst this
      rw [hnew]
      simp [x']
  · intro H₁ _ H₂ _ heq
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hup, hb⟩ := heq
    have hq : rq P H₁ u₁ = rq P H₂ u₁ := by rw [← hrq H₁, ← hrq H₂, hup]
    have hc : c1 chain8 H₁ P W0 u₁ = c1 chain8 H₂ P W0 u₁ := by
      have hx : x' H₁ = x' H₂ := by rw [← hnew H₁, ← hnew H₂, hup]
      funext c
      by_cases hc1 : c = 1
      · subst hc1; exact hb
      · have := congrFun hx c
        simpa [x', Function.update_of_ne hc1] using this
    have hval : ∀ H, H (rq P H u₁) = labXor (W0 u₁) (P.symm (c1 chain8 H P W0 u₁)) := by
      intro H
      rw [c1_eq]
      change H (rq P H u₁) = labXor (W0 u₁) (P.symm (P (labXor (W0 u₁) (H (rq P H u₁)))))
      rw [Equiv.symm_apply_apply, labXor_cancel]
    funext q
    by_cases hqq : q = rq P H₁ u₁
    · rw [hqq, hval H₁, hc, hq, hval H₂]
    · have h1 : upd H₁ q = H₁ q := Function.update_of_ne hqq _ _
      have h2 : upd H₂ q = H₂ q := Function.update_of_ne (hq ▸ hqq) _ _
      rw [← h1, ← h2, hup]

theorem card_Ev7 : Fintype.card (KQ 8 64 → Lab 8 64) ≤
    (Finset.univ.filter (Ev P 7)).card * Fintype.card (Bits 64) ^ 7 := by
  have h0 : (Finset.univ.filter (Ev P 0)).card = Fintype.card (KQ 8 64 → Lab 8 64) := by
    rw [Finset.filter_true_of_mem fun H _ u hu1 hu0 => absurd (hu1.trans hu0) (by omega), Finset.card_univ]
  have hstep : ∀ j, j ≤ 7 → (Finset.univ.filter (Ev P 0)).card ≤
      (Finset.univ.filter (Ev P j)).card * Fintype.card (Bits 64) ^ j := by
    intro j
    induction j with
    | zero => intro _; rw [pow_zero, mul_one]
    | succ j ih =>
      intro hj
      calc _ ≤ (Finset.univ.filter (Ev P j)).card * Fintype.card (Bits 64) ^ j := ih (by omega)
        _ ≤ (Finset.univ.filter (Ev P (j + 1))).card * Fintype.card (Bits 64) * Fintype.card (Bits 64) ^ j :=
            Nat.mul_le_mul_right _ (card_Ev_step P j (by omega))
        _ = _ := by rw [mul_assoc, ← pow_succ']
  rw [← h0]
  exact hstep 7 le_rfl

/-- For every permutation, the seven guesses are right with probability at least `2^-448` over the oracle. -/
theorem pr_Ev7 : (2⁻¹ : ℝ≥0∞) ^ 448 ≤ pr (Ev P 7) := by
  have hc := (Nat.cast_le (α := ℝ≥0∞)).mpr (card_Ev7 P)
  have h448 : 64 * 7 = 448 := by norm_num
  have hb : ((Fintype.card (Bits 64) : ℕ) : ℝ≥0∞) = 2 ^ 64 := by
    rw [Fintype.card_fun, Fintype.card_bool, Fintype.card_fin, Nat.cast_pow, Nat.cast_ofNat]
  rw [Nat.cast_mul, Nat.cast_pow (Fintype.card (Bits 64)), hb, ← pow_mul, h448] at hc
  have hΩ0 : (Fintype.card (KQ 8 64 → Lab 8 64) : ℝ≥0∞) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
  have hΩt : (Fintype.card (KQ 8 64 → Lab 8 64) : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  rw [pr_def, ENNReal.le_div_iff_mul_le (Or.inl hΩ0) (Or.inl hΩt)]
  calc (2⁻¹ : ℝ≥0∞) ^ 448 * Fintype.card (KQ 8 64 → Lab 8 64)
      ≤ (2⁻¹ : ℝ≥0∞) ^ 448 * (((Finset.univ.filter (Ev P 7)).card : ℝ≥0∞) * 2 ^ 448) := by gcongr
    _ = ((Finset.univ.filter (Ev P 7)).card : ℝ≥0∞) := by
        rw [mul_comm, mul_assoc, ← mul_pow, ENNReal.mul_inv_cancel two_ne_zero ENNReal.ofNat_ne_top, one_pow,
          mul_one]

end Count

/-! ### Over the whole primitive -/

/-- If every fibre over the second coordinate has probability at least `c`, so does the product event. -/
theorem pr_prod_ge {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β] (R : α → β → Prop) (c : ℝ≥0∞)
    (h : ∀ b, c ≤ pr (fun a => R a b)) : c ≤ pr (fun x : α × β => R x.1 x.2) := by
  classical
  have hα0 : (Fintype.card α : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hβ0 : (Fintype.card β : ℝ≥0∞) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hαt : (Fintype.card α : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hβt : (Fintype.card β : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hcount : ((Finset.univ.filter fun x : α × β => R x.1 x.2).card : ℝ≥0∞) =
      ∑ b, ((Finset.univ.filter fun a => R a b).card : ℝ≥0∞) := by
    rw [Finset.natCast_card_filter, Fintype.sum_prod_type, Finset.sum_comm]
    refine Finset.sum_congr rfl fun b _ => ?_
    rw [Finset.natCast_card_filter]
  rw [pr_def, hcount, Fintype.card_prod, Nat.cast_mul,
    ENNReal.le_div_iff_mul_le (Or.inl (mul_ne_zero hα0 hβ0)) (Or.inl (ENNReal.mul_ne_top hαt hβt))]
  calc c * (Fintype.card α * Fintype.card β) = ∑ _b : β, c * Fintype.card α := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; ring
    _ ≤ ∑ b, ((Finset.univ.filter fun a => R a b).card : ℝ≥0∞) := by
        refine Finset.sum_le_sum fun b _ => ?_
        have := h b
        rwa [pr_def, ENNReal.le_div_iff_mul_le (Or.inl hα0) (Or.inl hαt)] at this

theorem pr_congr_inst {α : Type} (i₁ i₂ : Fintype α) (P : α → Prop) : @pr α i₁ P = @pr α i₂ P := by
  rw [Subsingleton.elim i₁ i₂]

/-- `pr_prod_ge` for any `Fintype` instance on the product. -/
theorem pr_prod_ge' {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β] (i : Fintype (α × β))
    (R : α → β → Prop) (c : ℝ≥0∞) (h : ∀ b, c ≤ pr (fun a => R a b)) :
    c ≤ @pr (α × β) i (fun x => R x.1 x.2) :=
  (pr_prod_ge R c h).trans_eq (pr_congr_inst _ _ _)

/-- The adversary answers both challenges 0 and 1 with probability at least `2^-448`. -/
theorem pr_attack : (2⁻¹ : ℝ≥0∞) ^ 448 ≤ pr (fun ω : (p3Model 8 64).Ω =>
    2 ≤ (Finset.univ.filter fun v =>
      (A₂ (A₁ ω) v).run ((p3Model 8 64).answer ω) = topLabel chain8 W0 ω v).card) := by
  refine le_trans ?_ (pr_mono fun ω (hω : Ev ω.2 7 ω.1) => ?_)
  · exact pr_prod_ge' (α := KQ 8 64 → Lab 8 64) (β := Equiv.Perm (Lab 8 64)) (p3Model 8 64).fintypeΩ
      (fun H P => Ev P 7 H) _ (by intro P; exact pr_Ev7 P)
  · obtain ⟨H, P⟩ := ω
    exact correct_of H P hω

/-! ### The bound it beats -/

theorem bound_lt :
    ((8 : ℕ) : ℝ≥0∞) * 2 ^ 512 * ((((8 * (4 + 1) + 2 * 8) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (8 * 64)) ^ (1 + 1) <
      (2⁻¹ : ℝ≥0∞) ^ 448 := by
  have e1 : ((8 * (4 + 1) + 2 * 8) ^ 2 : ℕ) = 3136 := by norm_num
  have e2 : (8 * 64 : ℕ) = 512 := by norm_num
  rw [e1, e2, show (1 + 1 : ℕ) = 2 from rfl, Nat.cast_ofNat, Nat.cast_ofNat]
  have hy0 : (2⁻¹ : ℝ≥0∞) ^ 512 ≠ 0 := pow_ne_zero _ (ENNReal.inv_ne_zero.2 ENNReal.ofNat_ne_top)
  have hyt : (2⁻¹ : ℝ≥0∞) ^ 512 ≠ ⊤ := ENNReal.pow_ne_top (ENNReal.inv_ne_top.2 two_ne_zero)
  have hl : (8 : ℝ≥0∞) * 2 ^ 512 * ((3136 : ℝ≥0∞) / 2 ^ 512) ^ 2 = 78675968 * (2⁻¹ : ℝ≥0∞) ^ 512 := by
    rw [div_eq_mul_inv, mul_pow, ENNReal.inv_pow, ← pow_mul]
    calc (8 : ℝ≥0∞) * 2 ^ 512 * (3136 ^ 2 * (2⁻¹ : ℝ≥0∞) ^ (512 * 2))
        = (8 * 3136 ^ 2) * ((2 : ℝ≥0∞) ^ 512 * (2⁻¹ : ℝ≥0∞) ^ (512 * 2)) := by ring
      _ = 78675968 * (2⁻¹ : ℝ≥0∞) ^ 512 := by
          rw [two_pow_mul_inv_pow 512 (512 * 2) (by norm_num)]
          norm_num
  have hr : (2⁻¹ : ℝ≥0∞) ^ 448 = 2 ^ 64 * (2⁻¹ : ℝ≥0∞) ^ 512 := by
    rw [two_pow_mul_inv_pow 64 512 (by norm_num)]
  rw [hl, hr]
  exact ENNReal.mul_lt_mul_left hy0 hyt (by norm_num)

/-! ### The refutations -/

/-- **The pinned N10 is false.** -/
theorem not_invExPostFactoTwoLayer : ¬ Pous.Guarantees.InvExPostFactoTwoLayer := by
  intro h
  have hb := h 8 64 512 1 3 4 2 chain8 W0 (by norm_num) hard A₁ A₂ A₂_bounded
  exact absurd (pr_attack.trans hb) (not_le.2 bound_lt)

set_option linter.unusedVariables false in
/-- The `q_inv = 0` specialization of `pebbling/InvertibleExPostFactoDraft.lean`'s `inv_ex_post_facto_bounded_inv`,
verbatim with `q_inv := 0`. -/
def ForwardOnlyBound : Prop :=
  ∀ (n wc m s D Q k : ℕ) (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc)
    (hwc : 0 < wc) (hqs : 0 ≤ s)
    (hard : WhiteGreenHard G s D k)
    (A₁ : (p3Model n wc).Ω → Bits m)
    (A₂ : Bits m → Fin n → OracleProg (p3Model n wc) (Lab n wc))
    (hA₂ : ∀ σ c, (A₂ σ c).Bounded D Q)
    (hInv : ∀ σ c, InvBounded 0 (A₂ σ c)),
    pr (fun ω : (p3Model n wc).Ω =>
        k ≤ (Finset.univ.filter fun v => (A₂ (A₁ ω) v).run ((p3Model n wc).answer ω) = topLabel G W ω v).card) ≤
      (n : ℝ≥0∞) * 2 ^ m * ((((n * (Q + 1) + 2 * n) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (n * wc)) ^ (s + 1 - 0)

/-- **The forward-only case is false**, by the same adversary, which makes no inverse query. -/
theorem not_forwardOnlyBound : ¬ ForwardOnlyBound := by
  intro h
  have hb := h 8 64 512 1 3 4 2 chain8 W0 (by norm_num) (Nat.zero_le _) hard A₁ A₂ A₂_bounded A₂_invBounded
  exact absurd (pr_attack.trans hb) (not_le.2 bound_lt)

/-! ### The sibling's placeholder `ForwardBound` -/

set_option linter.unusedVariables false in
/-- `ForwardBound`, verbatim from `pebbling/InvertibleExPostFactoDraft.lean` (namespace `PousInvBoundedInv`). -/
def ForwardBound (n wc m D Q k : ℕ) (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc) : Prop :=
  ∀ (s' : ℕ) (A₁ : (p3Model n wc).Ω → Bits m) (A₂ : Bits m → Fin n → OracleProg (p3Model n wc) (Lab n wc)),
    (∀ σ c, (A₂ σ c).Bounded D Q) → (∀ σ c, InvBounded 0 (A₂ σ c)) → WhiteGreenHard G s' D k →
    pr (fun ω : (p3Model n wc).Ω =>
        k ≤ (Finset.univ.filter fun v => (A₂ (A₁ ω) v).run ((p3Model n wc).answer ω) = topLabel G W ω v).card) ≤
      (n : ℝ≥0∞) * 2 ^ m * ((((n * (Q + 1) + 2 * n) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ (n * wc)) ^ (s' + 1)

/-- **`ForwardBound` is false** at `n = 8, wc = 64, m = 512, D = 3, Q = 4, k = 2`, the chain, `W = 0` (with
`s' = 1`). -/
theorem not_forwardBound : ¬ ForwardBound 8 64 512 3 4 2 chain8 W0 := by
  intro h
  have hb := h 1 A₁ A₂ A₂_bounded A₂_invBounded hard
  exact absurd (pr_attack.trans hb) (not_le.2 bound_lt)

set_option linter.unusedVariables false in
/-- `ForwardBound` with the per-entry width `b` as a parameter: each of the `s' + 1` fresh entries saves `b` bits.
`ForwardBound` is `b = n·wc` (one whole label per fresh entry), refuted above. `b = wc` (one component per fresh
entry) is the weak candidate, open. -/
def ForwardBoundAt (b n wc m D Q k : ℕ) (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc) : Prop :=
  ∀ (s' : ℕ) (A₁ : (p3Model n wc).Ω → Bits m) (A₂ : Bits m → Fin n → OracleProg (p3Model n wc) (Lab n wc)),
    (∀ σ c, (A₂ σ c).Bounded D Q) → (∀ σ c, InvBounded 0 (A₂ σ c)) → WhiteGreenHard G s' D k →
    pr (fun ω : (p3Model n wc).Ω =>
        k ≤ (Finset.univ.filter fun v => (A₂ (A₁ ω) v).run ((p3Model n wc).answer ω) = topLabel G W ω v).card) ≤
      (n : ℝ≥0∞) * 2 ^ m * ((((n * (Q + 1) + 2 * n) ^ 2 : ℕ) : ℝ≥0∞) / 2 ^ b) ^ (s' + 1)

theorem forwardBound_eq_at (n wc m D Q k : ℕ) (G : Labelling.TopoDAG n) (W : Fin n → Lab n wc) :
    ForwardBound n wc m D Q k G W = ForwardBoundAt (n * wc) n wc m D Q k G W := rfl

end PousP3Forward
