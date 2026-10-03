import Pous.SecurityProofs.Thm1.Theorem1iTimed
import Pous.SecurityProofs.Thm1.Theorem1iiSeq
import Pous.SecurityProofs.Thm1.Theorem1iiiTimed
import Pous.SecurityProofs.Continuous.MissCount
import Pous.SecurityProofs.Continuous.Game

/-!
# The continuous audit from `TimedINC`: proofs of the drafts `PerAnswer` and `ContinuousAudit`

Not pinned: the statements are drafts (`Continuous/Game.lean`).

1. **One history** (`pr_bad_le`). For a history `h`, `bad h` is the event that the state after `h` and the wrong
   answers then fit in `β` bits, `|σ_h| + B + #wrong·ℓ ≤ β`. For each length `s ≤ β` of the state, Theorem 1(ii)'s
   patch compressor (`pr_fits_le`) bounds `|σ_h| = s ∧ bad h` by `π`, so `Pr[bad h] ≤ (β + 1)·π`.
2. **Step (a)** (`per_answer`). Off `bad h`, at most `p₀(β, |σ_h|) = 1 − ε + ε_proof` of the blocks are answered
   correctly (`good_le_p₀`).
3. **Stopping** (`continuous_audit`). Take `q_h = ε_h − ε_proof` off `bad h` and `0` on it. Then `q_h` lower-bounds
   the conditional miss probability at every setup and history, so the counting step (`pr_misses_gt_le`) bounds the
   misses against `Σ q` by `βe` at each setup. Off the event that some visited history is bad, `Σ q` is
   `Σ (ε_j − ε_proof)`. That event has probability at most `n·(β + 1)·π`: the indices are independent of the setup,
   so each round's history is bad with probability at most `(β + 1)·π`. The union runs over the `n` histories the
   audit visits, not over all `Σ_{j<n} B^j` of them as in `theorem1_ii_seq`.
-/

namespace PousContinuous

open Pous PousThm1
open scoped ENNReal

variable {M : IdealModel}

/-- `l` as an `s`-bit string, padded with `false` or truncated. -/
def toBits (s : ℕ) (l : List Bool) : Bits s := fun t => l.getD t false

theorem ofFn_toBits (s : ℕ) (l : List Bool) (h : l.length = s) : List.ofFn (toBits s l) = l := by
  subst h
  refine List.ext_getElem (by simp) fun t h1 h2 => ?_
  simp [toBits, List.getElem?_eq_getElem h2]

/-- **Averaging over the first coordinate.** -/
theorem pr_prod_le_of_forall {α β : Type} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (P : α → β → Prop) (c : ℝ≥0∞) (h : ∀ a, pr (P a) ≤ c) : pr (fun x : α × β => P x.1 x.2) ≤ c := by
  have := pr_prod_le P (fun _ => False) c fun a _ => h a
  rwa [pr_false', zero_add] at this

theorem pr_not_toReal {α : Type} [Fintype α] [Nonempty α] (P : α → Prop) :
    (pr fun x => ¬ P x).toReal = 1 - (pr P).toReal := by
  classical
  have h := Finset.card_filter_add_card_filter_not (s := Finset.univ) P
  rw [Finset.card_univ] at h
  have hα : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [pr_toReal, pr_toReal, eq_sub_iff_add_eq, ← add_div, div_eq_one_iff_eq hα.ne']
  exact_mod_cast (add_comm _ _).trans h

section

variable (sch : Scheme M) {D Q : ℕ} (W : Bits sch.n) (A₁ : VarPreprocessor sch) (R : VarResponder sch D Q)

open Classical in
/-- After `h`, the state and the wrong answers fit in `β` bits. -/
def bad (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) : Prop :=
  (stateAt sch W A₁ R a h).length + sch.B + (Finset.univ.filter fun i => ¬ hit sch W A₁ R a h i).card * sch.ℓ ≤ β

/-- **One history.** `Pr[bad h] ≤ (β + 1)·π`, by one patch compressor per length of the state. -/
theorem pr_bad_le (β : ℕ) (π : ℝ≥0∞) (hinc : TimedINC sch D Q β π) (h : List (Fin sch.B)) :
    pr (fun a => bad sch W A₁ R β a h) ≤ (β + 1) * π := by
  classical
  calc pr (fun a => bad sch W A₁ R β a h)
      ≤ pr (fun a => ∃ s ∈ Finset.range (β + 1),
          (stateAt sch W A₁ R a h).length = s ∧ bad sch W A₁ R β a h) :=
        pr_mono fun a ha => ⟨_, Finset.mem_range.2 (by unfold bad at ha; omega), rfl, ha⟩
    _ ≤ ∑ s ∈ Finset.range (β + 1),
          pr (fun a => (stateAt sch W A₁ R a h).length = s ∧ bad sch W A₁ R β a h) := pr_exists_le _ _
    _ ≤ ∑ _s ∈ Finset.range (β + 1), π := by
        refine Finset.sum_le_sum fun s _ => le_trans (pr_mono fun a ha => ?_)
          (pr_fits_le sch D Q β s π hinc W
            (fun W pp C ω => toBits s (varState (fun j σ i => R.upd W pp j σ i ω) 0 h (A₁ W pp C ω)))
            (fun W pp τ i => R.ans W pp h.length (List.ofFn τ) i) fun W pp τ i => R.bounded _ _ _ _ _)
        obtain ⟨hs, hb⟩ := ha
        have hof := ofFn_toBits s _ hs
        unfold bad hit at hb
        rw [hs] at hb
        unfold stateAt at hof hb
        simp only [hof]
        convert hb
    _ = (β + 1) * π := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, Nat.cast_add, Nat.cast_one]

/-- `p₀` at the state's size is `1 − ε + ε_proof`. -/
theorem p₀_eq (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) :
    p₀ β (stateAt sch W A₁ R a h).length sch.B sch.ℓ = 1 - freed sch W A₁ R a h + epsProof sch β := by
  unfold p₀ freed epsProof
  ring

/-- **Step (a) at one setup.** Off `bad h`, the challenge after `h` is answered correctly with probability at most
`1 − ε + ε_proof`. -/
theorem pr_hit_le (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) (hb : ¬ bad sch W A₁ R β a h) :
    pr (hit sch W A₁ R a h) ≤ ENNReal.ofReal (1 - freed sch W A₁ R a h + epsProof sch β) := by
  rw [← p₀_eq]
  exact pr_good_le_p₀ sch.B sch.ℓ β _ sch.B_pos _ hb

/-- Off `bad h`, the challenge after `h` misses with probability at least `ε − ε_proof`. -/
theorem miss_ge (β : ℕ) (a : M.Ω × sch.Coins) (h : List (Fin sch.B)) (hb : ¬ bad sch W A₁ R β a h) :
    freed sch W A₁ R a h - epsProof sch β ≤ (pr fun i => ¬ hit sch W A₁ R a h i).toReal := by
  classical
  have hgb := Finset.card_filter_add_card_filter_not (s := Finset.univ) (hit sch W A₁ R a h)
  rw [Finset.card_univ, Fintype.card_fin] at hgb
  have hb' : β < (stateAt sch W A₁ R a h).length + sch.B +
      (Finset.univ.filter fun i => ¬ hit sch W A₁ R a h i).card * sch.ℓ := by
    unfold bad at hb
    convert Nat.lt_of_not_le hb using 4
  have hr := good_le_p₀ sch.B sch.ℓ β _ _ _ sch.B_pos hgb hb'
  rw [p₀_eq] at hr
  have hB : (0 : ℝ) < sch.B := by exact_mod_cast sch.B_pos
  have hgb' : ((Finset.univ.filter (hit sch W A₁ R a h)).card : ℝ) +
      (Finset.univ.filter fun i => ¬ hit sch W A₁ R a h i).card = sch.B := by exact_mod_cast hgb
  rw [pr_toReal, Fintype.card_fin, le_div_iff₀ hB]
  nlinarith

end

/-- **Step (a)**, `PerAnswer`, from `TimedINC`. -/
theorem per_answer : PerAnswer := by
  intro M sch D Q β π hinc W A₁ R h
  exact le_trans (pr_mono fun a ha => by_contra fun hb => ha (pr_hit_le sch W A₁ R β a h hb))
    (pr_bad_le sch W A₁ R β π hinc h)

open Classical in
/-- **The continuous audit's guarantee**, `ContinuousAudit`, from `TimedINC`. -/
theorem continuous_audit : ContinuousAudit := by
  intro M sch D Q β n π hinc W A₁ R f βe hβe
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  set G := ⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ))) with hG
  let hist : (Fin n → Fin sch.B) → Fin n → List (Fin sch.B) := fun I j => (List.ofFn I).take j
  let ev : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ((Finset.univ.filter fun j : Fin n => ¬ hit sch W A₁ R a (hist I j) (I j)).card : ℝ) ≤ f ∧
      G < ∑ j : Fin n, (freed sch W A₁ R a (hist I j) - epsProof sch β)
  let q : M.Ω × sch.Coins → List (Fin sch.B) → ℝ := fun a h =>
    if bad sch W A₁ R β a h then 0 else freed sch W A₁ R a h - epsProof sch β
  let ev' : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ((Finset.univ.filter fun j : Fin n => ¬ hit sch W A₁ R a (hist I j) (I j)).card : ℝ) ≤ f ∧
      G < ∑ j : Fin n, q a (hist I j)
  let stop : M.Ω × sch.Coins → (Fin n → Fin sch.B) → Prop := fun a I =>
    ∃ j ∈ (Finset.univ : Finset (Fin n)), bad sch W A₁ R β a (hist I j)
  have hre : pr (fun x : Outcome sch n =>
      ((Finset.univ.filter fun j : Fin n =>
          ¬ hit sch W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) (x.2.2 j)).card : ℝ) ≤ f ∧
        G < ∑ j : Fin n, (freed sch W A₁ R (x.1, x.2.1) ((List.ofFn x.2.2).take j) - epsProof sch β)) =
      pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2) :=
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2)
  rw [hre]
  have hsplit : ∀ y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B), ev y.1 y.2 → stop y.1 y.2 ∨ ev' y.1 y.2 := by
    rintro ⟨a, I⟩ ⟨h1, h2⟩
    by_cases hs : stop a I
    · exact Or.inl hs
    · refine Or.inr ⟨h1, h2.trans_eq (Finset.sum_congr rfl fun j _ => ?_)⟩
      have : ¬ bad sch W A₁ R β a (hist I j) := fun hb => hs ⟨j, Finset.mem_univ _, hb⟩
      simp only [q, this, ↓reduceIte]
  have hstop : pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2) ≤ n * (β + 1) * π := by
    refine (pr_exists_le _ _).trans ?_
    calc ∑ j : Fin n, pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => bad sch W A₁ R β y.1 (hist y.2 j))
        ≤ ∑ _j : Fin n, ((β : ℝ≥0∞) + 1) * π := by
          refine Finset.sum_le_sum fun j _ => ?_
          rw [← pr_equiv (Equiv.prodComm _ _)]
          simp only [Equiv.prodComm_apply, Prod.fst_swap, Prod.snd_swap]
          exact pr_prod_le_of_forall (fun I a => bad sch W A₁ R β a (hist I j)) _
            fun I => pr_bad_le sch W A₁ R β π hinc (hist I j)
      _ = n * (β + 1) * π := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]
  have hmain : pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev' y.1 y.2) ≤ ENNReal.ofReal βe := by
    refine pr_prod_le_of_forall ev' _ fun a => ?_
    have hq : ∀ h : List (Fin sch.B), h.length < n →
        q a h ≤ (pr fun i => ¬ hit sch W A₁ R a h i).toReal := by
      intro h _
      by_cases hb : bad sch W A₁ R β a h
      · simp only [q, hb, ↓reduceIte]
        exact ENNReal.toReal_nonneg
      · simp only [q, hb, ↓reduceIte]
        exact miss_ge sch W A₁ R β a h hb
    convert pr_misses_gt_le n (fun h i => ¬ hit sch W A₁ R a h i) (q a) hq f βe hβe
  calc pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev y.1 y.2)
      ≤ pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2 ∨ ev' y.1 y.2) := pr_mono hsplit
    _ ≤ pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => stop y.1 y.2) +
          pr (fun y : (M.Ω × sch.Coins) × (Fin n → Fin sch.B) => ev' y.1 y.2) := pr_or_le _ _
    _ ≤ n * (β + 1) * π + ENNReal.ofReal βe := add_le_add hstop hmain
    _ = ENNReal.ofReal βe + n * (β + 1) * π := add_comm _ _

end PousContinuous
