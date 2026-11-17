module

public import Definitions.Core.Interaction
public import Definitions.Core.Rewinding
public import Definitions.Core.FirstHit
public import Proofs.Flock.Claims.Prob

/-! The first-hit finder: openings off the plurality table cost collisions, linearly. -/

@[expose] public section

open Finset

namespace Verity.FirstHit

open Verity.Rewinding Claims

theorem sum_tapes_succ {D A : Type} [Fintype D] [AddCommMonoid A] (K : ℕ) (f : D → (Fin K → D) → A) :
    ∑ ρ : Fin (K + 1) → D, f (ρ 0) (fun i => ρ i.succ) = ∑ d, ∑ σ : Fin K → D, f d σ :=
  ((Fintype.sum_equiv (Fin.consEquiv fun _ => D) (fun p : D × (Fin K → D) => f p.1 p.2)
    (fun ρ : Fin (K + 1) → D => f (ρ 0) fun i => ρ i.succ) fun p => by simp [Fin.consEquiv]).symm).trans
    (Fintype.sum_prod_type' f)

theorem sum_tapes_const {Ω : Type} [Fintype Ω] (K : ℕ) (c : ℝ) :
    ∑ _σ : Fin K → Ω, c = (Fintype.card Ω : ℝ) ^ K * c := by
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_pi_const, nsmul_eq_mul, Nat.cast_pow]

theorem sum_tapes_const_nat {Ω : Type} [Fintype Ω] (K : ℕ) (c : ℕ) :
    ∑ _σ : Fin K → Ω, c = Fintype.card Ω ^ K * c := by
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_pi_const, smul_eq_mul]

theorem indicator_nonneg (q : Prop) [Decidable q] : (0 : ℝ) ≤ if q then (1 : ℝ) else 0 := by
  split_ifs <;> norm_num

variable {Ω P V : Type}

/-- `o`, and `r` when `o` is no value. -/
def orHit : Option V → Option V → Option V
  | some v, _ => some v
  | none, r => r

/-- **The first hit**: the value at node `p` of the first of the tapes `ρ 0, ρ 1, …` that opens `p`. -/
def firstHit (op : Ω → P → Option V) (p : P) : (K : ℕ) → (Fin K → Ω) → Option V
  | 0, _ => none
  | K + 1, ρ => orHit (op (ρ 0) p) (firstHit op p K fun i => ρ i.succ)

/-- **The finder finds a pair** on the first tape `ω` and the further tapes `ρ`: at some node the first run opens, the
first further tape that opens it opens it to another value. -/
def Found (op : Ω → P → Option V) (K : ℕ) (ω : Ω) (ρ : Fin K → Ω) : Prop :=
  ∃ p v w, op ω p = some v ∧ firstHit op p K ρ = some w ∧ v ≠ w

/-- **The cap runs out**: none of the `K` further tapes opens some node the first run opens. -/
def Miss (op : Ω → P → Option V) (K : ℕ) (ω : Ω) (ρ : Fin K → Ω) : Prop :=
  ∃ p, (op ω p).isSome ∧ firstHit op p K ρ = none

/-- A node's score against the first run's value `v`: `2` when the first hit is another value, `1` when there is no
hit, `0` when the first hit is `v` again. -/
noncomputable def score [DecidableEq V] (v : V) : Option V → ℝ
  | none => 1
  | some w => if w = v then 0 else 2

theorem sum_firstHit_succ {A : Type} [AddCommMonoid A] [Fintype Ω] (op : Ω → P → Option V) (p : P) (K : ℕ)
    (G : Option V → A) :
    ∑ ρ : Fin (K + 1) → Ω, G (firstHit op p (K + 1) ρ) =
      ∑ ω, ∑ σ : Fin K → Ω, G (orHit (op ω p) (firstHit op p K σ)) :=
  (Finset.sum_congr rfl fun _ _ => rfl).trans
    (sum_tapes_succ K fun (ω : Ω) (σ : Fin K → Ω) => G (orHit (op ω p) (firstHit op p K σ)))

theorem two_mul_openedTo_le [Fintype Ω] [DecidableEq V] (op : Ω → P → Option V) {vstar : P → V}
    (hv : IsPlurality op vstar) {p : P} {v : V} (hne : v ≠ vstar p) :
    2 * #(openedTo op p v) ≤ #(opened op p) := by
  have h1 : #(openedTo op p v) = ∑ ω, if op ω p = some v then 1 else 0 := Finset.card_filter _ _
  have h2 : #(openedTo op p (vstar p)) = ∑ ω, if op ω p = some (vstar p) then 1 else 0 :=
    Finset.card_filter _ _
  have h3 : #(opened op p) = ∑ ω, if (op ω p).isSome then 1 else 0 := Finset.card_filter _ _
  have hpt : ∀ ω : Ω, (if op ω p = some v then 1 else 0) + (if op ω p = some (vstar p) then 1 else 0) ≤
      (if (op ω p).isSome then 1 else 0) := by
    intro ω
    cases op ω p with
    | none => simp
    | some w =>
      by_cases hw : w = v
      · subst hw
        simp [hne]
      · by_cases hw' : w = vstar p
        · subst hw'
          simp [hw]
        · simp [hw, hw']
  have hs := Finset.sum_le_sum (s := (univ : Finset Ω)) fun ω _ => hpt ω
  rw [Finset.sum_add_distrib, ← h1, ← h2, ← h3] at hs
  have hp := hv p v
  omega

theorem card_le_sum_score [Fintype Ω] [DecidableEq V] (op : Ω → P → Option V) {p : P} {v : V}
    (h : 2 * #(openedTo op p v) ≤ #(opened op p)) :
    (Fintype.card Ω : ℝ) ≤ ∑ ω, score v (op ω p) := by
  have h1 : (#(openedTo op p v) : ℝ) = ∑ ω, if op ω p = some v then (1 : ℝ) else 0 :=
    Finset.natCast_card_filter _ _
  have h2 : (#(opened op p) : ℝ) = ∑ ω, if (op ω p).isSome then (1 : ℝ) else 0 :=
    Finset.natCast_card_filter _ _
  have hle : (2 : ℝ) * #(openedTo op p v) ≤ #(opened op p) := by exact_mod_cast h
  have hpt : ∀ ω : Ω, score v (op ω p) + 2 * (if op ω p = some v then (1 : ℝ) else 0) =
      1 + (if (op ω p).isSome then (1 : ℝ) else 0) := by
    intro ω
    cases op ω p with
    | none => simp [score]
    | some w =>
      by_cases hw : w = v
      · simp [score, hw]
        norm_num
      · simp [score, hw]
        norm_num
  have e : ∑ ω, (score v (op ω p) + 2 * (if op ω p = some v then (1 : ℝ) else 0)) =
      ∑ ω : Ω, ((1 : ℝ) + (if (op ω p).isSome then (1 : ℝ) else 0)) :=
    Finset.sum_congr rfl fun ω _ => hpt ω
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one, ← h1, ← h2] at e
  linarith

theorem pow_le_sum_score [Fintype Ω] [DecidableEq V] (op : Ω → P → Option V) {p : P} {v : V}
    (h : 2 * #(openedTo op p v) ≤ #(opened op p)) (K : ℕ) :
    (Fintype.card Ω : ℝ) ^ K ≤ ∑ ρ : Fin K → Ω, score v (firstHit op p K ρ) := by
  induction K with
  | zero =>
    have e : ∑ ρ : Fin 0 → Ω, score v (firstHit op p 0 ρ) = ∑ _ρ : Fin 0 → Ω, (1 : ℝ) :=
      Finset.sum_congr rfl fun _ _ => rfl
    refine le_of_eq ?_
    rw [e, sum_tapes_const, mul_one]
  | succ K ih =>
    rw [sum_firstHit_succ op p K (score v)]
    calc (Fintype.card Ω : ℝ) ^ (K + 1) = (Fintype.card Ω : ℝ) ^ K * Fintype.card Ω := pow_succ _ _
      _ ≤ (Fintype.card Ω : ℝ) ^ K * ∑ ω, score v (op ω p) :=
        mul_le_mul_of_nonneg_left (card_le_sum_score op h) (pow_nonneg (Nat.cast_nonneg _) K)
      _ = ∑ ω, (Fintype.card Ω : ℝ) ^ K * score v (op ω p) := Finset.mul_sum _ _ _
      _ ≤ ∑ ω, ∑ σ : Fin K → Ω, score v (orHit (op ω p) (firstHit op p K σ)) :=
        Finset.sum_le_sum fun ω _ => by
          cases op ω p with
          | none => exact (mul_one _).trans_le ih
          | some w => exact (sum_tapes_const K (score v (some w))).ge

theorem score_le [DecidableEq V] (op : Ω → P → Option V) (K : ℕ) {ω : Ω} {p : P} {v : V} (h : op ω p = some v)
    (ρ : Fin K → Ω) [Decidable (Found op K ω ρ)] [Decidable (Miss op K ω ρ)] :
    score v (firstHit op p K ρ) ≤
      2 * (if Found op K ω ρ then (1 : ℝ) else 0) + (if Miss op K ω ρ then (1 : ℝ) else 0) := by
  have hF : (0 : ℝ) ≤ if Found op K ω ρ then (1 : ℝ) else 0 := indicator_nonneg _
  have hZ : (0 : ℝ) ≤ if Miss op K ω ρ then (1 : ℝ) else 0 := indicator_nonneg _
  cases hf : firstHit op p K ρ with
  | none =>
    have hm : Miss op K ω ρ := Exists.intro p ⟨by simp [h], hf⟩
    have h1 : score v (none : Option V) = 1 := rfl
    rw [ite_eq_left hm, h1]
    linarith
  | some w =>
    by_cases hw : w = v
    · have h0 : score v (some w) = 0 := by simp [score, hw]
      rw [h0]
      linarith
    · have hfound : Found op K ω ρ := Exists.intro p (Exists.intro v (Exists.intro w ⟨h, hf, fun e => hw e.symm⟩))
      have h2 : score v (some w) = 2 := by simp [score, hw]
      rw [ite_eq_left hfound, h2]
      linarith

theorem off_mul_pow_le [Fintype Ω] [DecidableEq V] (op : Ω → P → Option V) {vstar : P → V}
    (hv : IsPlurality op vstar) (K : ℕ) (ω : Ω) [Decidable (∃ p, Off op vstar ω p)]
    [DecidablePred (Found op K ω)] [DecidablePred (Miss op K ω)] :
    (if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) * (Fintype.card Ω : ℝ) ^ K ≤
      ∑ ρ : Fin K → Ω,
        (2 * (if Found op K ω ρ then (1 : ℝ) else 0) + (if Miss op K ω ρ then (1 : ℝ) else 0)) := by
  by_cases hb : ∃ p, Off op vstar ω p
  · rw [ite_eq_left hb, one_mul]
    obtain ⟨p, hs, hne⟩ := hb
    obtain ⟨v, hv'⟩ := Option.isSome_iff_exists.1 hs
    have hvne : v ≠ vstar p := fun e => hne (by rw [hv', e])
    exact (pow_le_sum_score op (two_mul_openedTo_le op hv hvne) K).trans
      (Finset.sum_le_sum fun ρ _ => score_le op K hv' ρ)
  · rw [ite_eq_right hb, zero_mul]
    exact Finset.sum_nonneg fun ρ _ =>
      add_nonneg (mul_nonneg zero_le_two (indicator_nonneg _)) (indicator_nonneg _)

theorem pr_off_le [Fintype Ω] [Nonempty Ω] [DecidableEq V] (op : Ω → P → Option V) {vstar : P → V}
    (hv : IsPlurality op vstar) (K : ℕ) :
    Pr (fun ω => ∃ p, Off op vstar ω p) ≤
      2 * Pr (fun s : Ω × (Fin K → Ω) => Found op K s.1 s.2) +
        Pr (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2) := by
  classical
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hMK : (0 : ℝ) < (Fintype.card Ω : ℝ) ^ K := pow_pos hM K
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow]
  have eA : Pr (fun ω => ∃ p, Off op vstar ω p) =
      (∑ ω, if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) / Fintype.card Ω := Pr_eq_sum _
  have eF : Pr (fun s : Ω × (Fin K → Ω) => Found op K s.1 s.2) =
      (∑ ω, ∑ ρ : Fin K → Ω, if Found op K ω ρ then (1 : ℝ) else 0) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) := by
    rw [Pr_eq_sum (fun s : Ω × (Fin K → Ω) => Found op K s.1 s.2), hcard]
    exact congrArg (fun t : ℝ => t / ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K))
      (Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) => if Found op K ω ρ then (1 : ℝ) else 0)
  have eZ : Pr (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2) =
      (∑ ω, ∑ ρ : Fin K → Ω, if Miss op K ω ρ then (1 : ℝ) else 0) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) := by
    rw [Pr_eq_sum (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2), hcard]
    exact congrArg (fun t : ℝ => t / ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K))
      (Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) => if Miss op K ω ρ then (1 : ℝ) else 0)
  have key : (∑ ω, if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) * (Fintype.card Ω : ℝ) ^ K ≤
      2 * (∑ ω, ∑ ρ : Fin K → Ω, if Found op K ω ρ then (1 : ℝ) else 0) +
        ∑ ω, ∑ ρ : Fin K → Ω, if Miss op K ω ρ then (1 : ℝ) else 0 := by
    rw [Finset.sum_mul, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun ω _ => ?_
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact off_mul_pow_le op hv K ω
  rw [eA, eF, eZ]
  calc (∑ ω, if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) / (Fintype.card Ω : ℝ)
      = ((∑ ω, if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) * (Fintype.card Ω : ℝ) ^ K) /
          ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) := (mul_div_mul_right _ _ hMK.ne').symm
    _ ≤ (2 * (∑ ω, ∑ ρ : Fin K → Ω, if Found op K ω ρ then (1 : ℝ) else 0) +
          ∑ ω, ∑ ρ : Fin K → Ω, if Miss op K ω ρ then (1 : ℝ) else 0) /
          ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) :=
        div_le_div_of_nonneg_right key (mul_pos hM hMK).le
    _ = _ := by ring

/-- **The further runs' calls**: each further tape's, in order, while some node of `need` has not been opened again. -/
def loopCost (op : Ω → P → Option V) (c : Ω → ℕ) : (K : ℕ) → Finset P → (Fin K → Ω) → ℕ
  | 0, _, _ => 0
  | K + 1, need, ρ =>
    if need.Nonempty then
      c (ρ 0) + loopCost op c K (need.filter fun p => (op (ρ 0) p).isNone) fun i => ρ i.succ
    else 0

/-- The further tapes' calls up to and including the first that opens `p`. -/
def nodeCost (op : Ω → P → Option V) (c : Ω → ℕ) (p : P) : (K : ℕ) → (Fin K → Ω) → ℕ
  | 0, _ => 0
  | K + 1, ρ => c (ρ 0) + if (op (ρ 0) p).isNone then nodeCost op c p K (fun i => ρ i.succ) else 0

theorem loopCost_le (op : Ω → P → Option V) (c : Ω → ℕ) (K : ℕ) :
    ∀ (need : Finset P) (ρ : Fin K → Ω), loopCost op c K need ρ ≤ ∑ p ∈ need, nodeCost op c p K ρ := by
  induction K with
  | zero =>
    intro need ρ
    exact Nat.zero_le _
  | succ K ih =>
    intro need ρ
    by_cases hn : need.Nonempty
    · have e : loopCost op c (K + 1) need ρ =
          c (ρ 0) + loopCost op c K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := by
        simp only [loopCost, hn, ↓reduceIte]
      have e3 : ∑ p ∈ need, nodeCost op c p (K + 1) ρ =
          ∑ p ∈ need, (c (ρ 0) + if (op (ρ 0) p).isNone then nodeCost op c p K (fun i => ρ i.succ) else 0) :=
        Finset.sum_congr rfl fun _ _ => rfl
      have e2 : ∑ p ∈ need, nodeCost op c p (K + 1) ρ =
          #need * c (ρ 0) +
            ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone), nodeCost op c p K (fun i => ρ i.succ) := by
        rw [e3, Finset.sum_add_distrib, Finset.sum_const, smul_eq_mul, Finset.sum_filter]
      have h1 : 1 ≤ #need := Finset.card_pos.2 hn
      rw [e, e2]
      calc c (ρ 0) + loopCost op c K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ)
          ≤ c (ρ 0) +
              ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone), nodeCost op c p K (fun i => ρ i.succ) :=
            Nat.add_le_add_left (ih _ _) _
        _ = 1 * c (ρ 0) +
              ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone), nodeCost op c p K (fun i => ρ i.succ) := by
            rw [one_mul]
        _ ≤ #need * c (ρ 0) +
              ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone), nodeCost op c p K (fun i => ρ i.succ) :=
            Nat.add_le_add_right (Nat.mul_le_mul_right _ h1) _
    · have e : loopCost op c (K + 1) need ρ = 0 := by
        simp only [loopCost, hn, ↓reduceIte]
      rw [e]
      exact Nat.zero_le _

theorem card_opened_add [Fintype Ω] (op : Ω → P → Option V) (p : P) :
    #(opened op p) + #(univ.filter fun ω : Ω => (op ω p).isNone) = Fintype.card Ω := by
  rw [opened, Finset.card_filter, Finset.card_filter, ← Finset.sum_add_distrib, Fintype.card_eq_sum_ones]
  refine Finset.sum_congr rfl fun ω _ => ?_
  cases op ω p <;> simp

theorem nat_step {a b m k C N : ℕ} (hab : a + b = m) (ih : a * N ≤ k * C) :
    a * (k * C + b * N) ≤ k * m * C := by
  calc a * (k * C + b * N) = a * (k * C) + b * (a * N) := by ring
    _ ≤ a * (k * C) + b * (k * C) := Nat.add_le_add_left (Nat.mul_le_mul_left _ ih) _
    _ = k * m * C := by rw [← hab]; ring

theorem sum_nodeCost_succ [Fintype Ω] (op : Ω → P → Option V) (c : Ω → ℕ) (p : P) (K : ℕ) :
    ∑ ρ : Fin (K + 1) → Ω, nodeCost op c p (K + 1) ρ =
      Fintype.card Ω ^ K * ∑ ω, c ω +
        #(univ.filter fun ω : Ω => (op ω p).isNone) * ∑ σ : Fin K → Ω, nodeCost op c p K σ := by
  have e : ∑ ρ : Fin (K + 1) → Ω, nodeCost op c p (K + 1) ρ =
      ∑ ω : Ω, ∑ σ : Fin K → Ω, (c ω + if (op ω p).isNone then nodeCost op c p K σ else 0) :=
    (Finset.sum_congr rfl fun _ _ => rfl).trans
      (sum_tapes_succ K fun (ω : Ω) (σ : Fin K → Ω) => c ω + if (op ω p).isNone then nodeCost op c p K σ else 0)
  rw [e, Finset.mul_sum, Finset.card_filter, Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun ω _ => ?_
  rw [Finset.sum_add_distrib, sum_tapes_const_nat]
  cases op ω p <;> simp

theorem mul_sum_nodeCost_le [Fintype Ω] (op : Ω → P → Option V) (c : Ω → ℕ) (p : P) (K : ℕ) :
    #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op c p K ρ ≤ Fintype.card Ω ^ K * ∑ ω, c ω := by
  induction K with
  | zero =>
    have e : ∑ ρ : Fin 0 → Ω, nodeCost op c p 0 ρ = 0 := Finset.sum_eq_zero fun _ _ => rfl
    rw [e, Nat.mul_zero]
    exact Nat.zero_le _
  | succ K ih =>
    rw [sum_nodeCost_succ, pow_succ]
    exact nat_step (card_opened_add op p) ih

theorem sum_need [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (g : P → ℕ) :
    ∑ ω : Ω, ∑ p ∈ univ.filter (fun p => (op ω p).isSome), g p = ∑ p, #(opened op p) * g p :=
  calc ∑ ω : Ω, ∑ p ∈ univ.filter (fun p => (op ω p).isSome), g p
      = ∑ ω : Ω, ∑ p, if (op ω p).isSome then g p else 0 :=
        Finset.sum_congr rfl fun ω _ => Finset.sum_filter _ _
    _ = ∑ p, ∑ ω : Ω, if (op ω p).isSome then g p else 0 := Finset.sum_comm
    _ = ∑ p, #(opened op p) * g p := Finset.sum_congr rfl fun p _ =>
        (by rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul] :
          (∑ ω : Ω, if (op ω p).isSome then g p else 0) = #(univ.filter fun ω : Ω => (op ω p).isSome) * g p)

theorem sum_cost_le [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (c : Ω → ℕ) (K : ℕ) :
    ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCost op c K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      (1 + #(univ.filter fun p : P => (opened op p).Nonempty)) * (Fintype.card Ω ^ K * ∑ ω, c ω) := by
  have h1 : ∀ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCost op c K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      Fintype.card Ω ^ K * c ω +
        ∑ p ∈ univ.filter (fun p => (op ω p).isSome), ∑ ρ : Fin K → Ω, nodeCost op c p K ρ := by
    intro ω
    rw [Finset.sum_add_distrib, sum_tapes_const_nat, Finset.sum_comm]
    exact Nat.add_le_add_left (Finset.sum_le_sum fun ρ _ => loopCost_le op c K _ ρ) _
  have h2 : ∀ p : P, #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op c p K ρ ≤
      if (opened op p).Nonempty then Fintype.card Ω ^ K * ∑ ω, c ω else 0 := by
    intro p
    by_cases hp : (opened op p).Nonempty
    · rw [ite_eq_left hp]
      exact mul_sum_nodeCost_le op c p K
    · have h0 : #(opened op p) = 0 := Finset.card_eq_zero.2 (Finset.not_nonempty_iff_eq_empty.1 hp)
      exact le_of_eq (by rw [ite_eq_right hp, h0, Nat.zero_mul])
  calc ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCost op c K (univ.filter fun p => (op ω p).isSome) ρ)
      ≤ ∑ ω : Ω, (Fintype.card Ω ^ K * c ω +
          ∑ p ∈ univ.filter (fun p => (op ω p).isSome), ∑ ρ : Fin K → Ω, nodeCost op c p K ρ) :=
        Finset.sum_le_sum fun ω _ => h1 ω
    _ = Fintype.card Ω ^ K * ∑ ω, c ω + ∑ p, #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op c p K ρ := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_need op]
    _ ≤ Fintype.card Ω ^ K * ∑ ω, c ω +
          ∑ p, (if (opened op p).Nonempty then Fintype.card Ω ^ K * ∑ ω, c ω else 0) :=
        Nat.add_le_add_left (Finset.sum_le_sum fun p _ => h2 p) _
    _ = (1 + #(univ.filter fun p : P => (opened op p).Nonempty)) * (Fintype.card Ω ^ K * ∑ ω, c ω) := by
        rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
        ring

theorem ex_cost_le [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (c : Ω → ℕ) (K : ℕ) :
    Ex (fun s : Ω × (Fin K → Ω) =>
        ((c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) * Ex (fun ω : Ω => (c ω : ℝ)) := by
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hMK : (0 : ℝ) < (Fintype.card Ω : ℝ) ^ K := pow_pos hM K
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow]
  have hN : ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2) ≤
      (1 + #(univ.filter fun p : P => (opened op p).Nonempty)) * (Fintype.card Ω ^ K * ∑ ω, c ω) :=
    (Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) =>
      c ω + loopCost op c K (univ.filter fun p => (op ω p).isSome) ρ).trans_le (sum_cost_le op c K)
  have hR : (∑ s : Ω × (Fin K → Ω),
        ((c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) *
        ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ)) := by
    exact_mod_cast hN
  unfold Ex
  rw [hcard]
  calc (∑ s : Ω × (Fin K → Ω),
          ((c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K)
      ≤ ((1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) *
          ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ))) / ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) :=
        div_le_div_of_nonneg_right hR (mul_pos hM hMK).le
    _ = (1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) *
          ((∑ ω, (c ω : ℝ)) / (Fintype.card Ω : ℝ)) := by
        rw [show (1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) *
            ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ)) =
            ((1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) * ∑ ω, (c ω : ℝ)) *
              (Fintype.card Ω : ℝ) ^ K by ring,
          mul_div_mul_right _ _ hMK.ne', mul_div_assoc]

/-- **The first-hit finder** as a collision finder: its coins are a first tape and `K` further tapes, its output `out`,
and its cost the calls of the runs it makes: the first tape's, and each further tape's while some node the first run
opened has not been opened again. -/
noncomputable def finder [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (c : Ω → ℕ) (K : ℕ)
    (out : Ω × (Fin K → Ω) → Option (Msg × Msg)) : Finder where
  S := Ω × (Fin K → Ω)
  out := out
  cost s := c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2

theorem pr_off_le_of_expectedCR [Fintype Ω] [Nonempty Ω] [Fintype P] [DecidableEq V] (H : Msg → Msg)
    (op : Ω → P → Option V) {vstar : P → V} (hv : IsPlurality op vstar) (c : Ω → ℕ) (K : ℕ)
    (out : Ω × (Fin K → Ω) → Option (Msg × Msg))
    (hout : ∀ s : Ω × (Fin K → Ω), Found op K s.1 s.2 → ∃ u v, out s = some (u, v) ∧ Collision H u v)
    (hCR : ExpectedCR H (finder op c K out)) :
    Pr (fun ω => ∃ p, Off op vstar ω p) ≤
      2 * ((1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) * Ex (fun ω : Ω => (c ω : ℝ))) /
          2 ^ (256 : ℝ) +
        Pr (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2) := by
  have h1 := pr_off_le op hv K
  have h2 : Pr (fun s : Ω × (Fin K → Ω) => Found op K s.1 s.2) ≤
      Pr (fun s : Ω × (Fin K → Ω) => ∃ u v, out s = some (u, v) ∧ Collision H u v) := Pr_mono hout
  have h3 : Pr (fun s : Ω × (Fin K → Ω) => ∃ u v, out s = some (u, v) ∧ Collision H u v) ≤
      Ex (fun s : Ω × (Fin K → Ω) =>
        ((c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) / 2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right (ex_cost_le op c K) (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  rw [mul_div_assoc]
  linarith

theorem sum_firstHit_isNone [Fintype Ω] (op : Ω → P → Option V) (p : P) (K : ℕ) :
    ∑ ρ : Fin K → Ω, (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) =
      #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K := by
  induction K with
  | zero =>
    have e : ∑ ρ : Fin 0 → Ω, (if (firstHit op p 0 ρ).isNone then (1 : ℕ) else 0) = ∑ _ρ : Fin 0 → Ω, 1 :=
      Finset.sum_congr rfl fun _ _ => by simp [firstHit]
    rw [e, sum_tapes_const_nat]
    simp
  | succ K ih =>
    have h : ∀ ω : Ω,
        ∑ σ : Fin K → Ω, (if (orHit (op ω p) (firstHit op p K σ)).isNone then (1 : ℕ) else 0) =
          if (op ω p).isNone then #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K else 0 := by
      intro ω
      cases op ω p with
      | none => simpa [orHit] using ih
      | some w => simp [orHit]
    calc ∑ ρ : Fin (K + 1) → Ω, (if (firstHit op p (K + 1) ρ).isNone then (1 : ℕ) else 0)
        = ∑ ω, ∑ σ : Fin K → Ω, (if (orHit (op ω p) (firstHit op p K σ)).isNone then (1 : ℕ) else 0) :=
          sum_firstHit_succ op p K fun o : Option V => if o.isNone then (1 : ℕ) else 0
      _ = ∑ ω, if (op ω p).isNone then #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K else 0 :=
          Finset.sum_congr rfl fun ω _ => h ω
      _ = #(univ.filter fun ω : Ω => (op ω p).isNone) ^ (K + 1) := by
          rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul, pow_succ']

theorem miss_le_sum [Fintype P] (op : Ω → P → Option V) (K : ℕ) (ω : Ω) (ρ : Fin K → Ω)
    [Decidable (Miss op K ω ρ)] :
    (if Miss op K ω ρ then (1 : ℕ) else 0) ≤
      ∑ p, (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) := by
  by_cases hm : Miss op K ω ρ
  · rw [ite_eq_left hm]
    obtain ⟨p, hs, hn⟩ := hm
    have hp : (1 : ℕ) ≤
        (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) :=
      le_of_eq (by rw [ite_eq_left hs, ite_eq_left (Option.isNone_iff_eq_none.2 hn), Nat.mul_one])
    exact hp.trans (Finset.single_le_sum
      (f := fun p => (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0))
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ p))
  · rw [ite_eq_right hm]
    exact Nat.zero_le _

theorem succ_mul_mul_pow_le (a b K : ℕ) : (K + 1) * (a * b ^ K) ≤ (a + b) ^ (K + 1) := by
  induction K with
  | zero => simp
  | succ K ih =>
    have hb : b ^ (K + 1) ≤ (a + b) ^ (K + 1) := Nat.pow_le_pow_left (Nat.le_add_left b a) (K + 1)
    calc (K + 1 + 1) * (a * b ^ (K + 1)) = a * b ^ (K + 1) + b * ((K + 1) * (a * b ^ K)) := by ring
      _ ≤ a * (a + b) ^ (K + 1) + b * (a + b) ^ (K + 1) :=
        Nat.add_le_add (Nat.mul_le_mul_left _ hb) (Nat.mul_le_mul_left _ ih)
      _ = (a + b) ^ (K + 1 + 1) := by ring

theorem succ_mul_sum_miss_le [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (K : ℕ)
    [∀ (ω : Ω) (ρ : Fin K → Ω), Decidable (Miss op K ω ρ)] :
    (K + 1) * ∑ ω : Ω, ∑ ρ : Fin K → Ω, (if Miss op K ω ρ then (1 : ℕ) else 0) ≤
      #(univ.filter fun p : P => (opened op p).Nonempty) * Fintype.card Ω ^ (K + 1) := by
  have h1 : ∑ ω : Ω, ∑ ρ : Fin K → Ω, (if Miss op K ω ρ then (1 : ℕ) else 0) ≤
      ∑ p, #(opened op p) * #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K :=
    calc ∑ ω : Ω, ∑ ρ : Fin K → Ω, (if Miss op K ω ρ then (1 : ℕ) else 0)
        ≤ ∑ ω : Ω, ∑ ρ : Fin K → Ω, ∑ p,
            (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) :=
          Finset.sum_le_sum fun ω _ => Finset.sum_le_sum fun ρ _ => miss_le_sum op K ω ρ
      _ = ∑ ω : Ω, ∑ p, ∑ ρ : Fin K → Ω,
            (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) :=
          Finset.sum_congr rfl fun ω _ => Finset.sum_comm
      _ = ∑ p, ∑ ω : Ω, ∑ ρ : Fin K → Ω,
            (if (op ω p).isSome then (1 : ℕ) else 0) * (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) :=
          Finset.sum_comm
      _ = ∑ p, (∑ ω : Ω, if (op ω p).isSome then (1 : ℕ) else 0) *
            ∑ ρ : Fin K → Ω, (if (firstHit op p K ρ).isNone then (1 : ℕ) else 0) :=
          Finset.sum_congr rfl fun p _ => (Finset.sum_mul_sum _ _ _ _).symm
      _ = ∑ p, #(opened op p) * #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K :=
          Finset.sum_congr rfl fun p _ => by rw [sum_firstHit_isNone, ← Finset.card_filter, opened]
  have h2 : ∀ p : P, (K + 1) * (#(opened op p) * #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K) ≤
      if (opened op p).Nonempty then Fintype.card Ω ^ (K + 1) else 0 := by
    intro p
    by_cases hp : (opened op p).Nonempty
    · rw [ite_eq_left hp, ← card_opened_add op p]
      exact succ_mul_mul_pow_le _ _ K
    · have h0 : #(opened op p) = 0 := Finset.card_eq_zero.2 (Finset.not_nonempty_iff_eq_empty.1 hp)
      exact le_of_eq (by rw [ite_eq_right hp, h0, Nat.zero_mul, Nat.mul_zero])
  calc (K + 1) * ∑ ω : Ω, ∑ ρ : Fin K → Ω, (if Miss op K ω ρ then (1 : ℕ) else 0)
      ≤ (K + 1) * ∑ p, #(opened op p) * #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K :=
        Nat.mul_le_mul_left _ h1
    _ = ∑ p, (K + 1) * (#(opened op p) * #(univ.filter fun ω : Ω => (op ω p).isNone) ^ K) :=
        Finset.mul_sum _ _ _
    _ ≤ ∑ p, (if (opened op p).Nonempty then Fintype.card Ω ^ (K + 1) else 0) :=
        Finset.sum_le_sum fun p _ => h2 p
    _ = #(univ.filter fun p : P => (opened op p).Nonempty) * Fintype.card Ω ^ (K + 1) := by
        rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]

theorem pr_miss_le [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (K : ℕ) :
    Pr (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2) ≤
      (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ) / ((K : ℝ) + 1) := by
  classical
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hK : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) ^ (K + 1) := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow, pow_succ']
  have eZ : Pr (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2) =
      (∑ ω, ∑ ρ : Fin K → Ω, if Miss op K ω ρ then (1 : ℝ) else 0) / (Fintype.card Ω : ℝ) ^ (K + 1) := by
    rw [Pr_eq_sum (fun s : Ω × (Fin K → Ω) => Miss op K s.1 s.2), hcard]
    exact congrArg (fun t : ℝ => t / (Fintype.card Ω : ℝ) ^ (K + 1))
      (Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) => if Miss op K ω ρ then (1 : ℝ) else 0)
  have hN := succ_mul_sum_miss_le op K
  have hR : ((K : ℝ) + 1) * ∑ ω : Ω, ∑ ρ : Fin K → Ω, (if Miss op K ω ρ then (1 : ℝ) else 0) ≤
      (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ) * (Fintype.card Ω : ℝ) ^ (K + 1) := by
    exact_mod_cast hN
  rw [eZ, div_le_div_iff₀ (pow_pos hM (K + 1)) hK]
  exact (mul_comm _ _).trans_le hR

theorem card_opened_nonempty_le [Fintype Ω] [Fintype P] (op : Ω → P → Option V) :
    #(univ.filter fun p : P => (opened op p).Nonempty) ≤ Fintype.card P :=
  Finset.card_le_univ _

theorem pr_off_le_of_expectedCR_cap [Fintype Ω] [Nonempty Ω] [Fintype P] [DecidableEq V] (H : Msg → Msg)
    (op : Ω → P → Option V) {vstar : P → V} (hv : IsPlurality op vstar) (c : Ω → ℕ) (K : ℕ)
    (out : Ω × (Fin K → Ω) → Option (Msg × Msg))
    (hout : ∀ s : Ω × (Fin K → Ω), Found op K s.1 s.2 → ∃ u v, out s = some (u, v) ∧ Collision H u v)
    (hCR : ExpectedCR H (finder op c K out)) :
    Pr (fun ω => ∃ p, Off op vstar ω p) ≤
      2 * ((1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) * Ex (fun ω : Ω => (c ω : ℝ))) /
          2 ^ (256 : ℝ) +
        (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_le_of_expectedCR H op hv c K out hout hCR
  have h2 := pr_miss_le op K
  linarith

theorem pr_off_le_of_forall_expectedCR [Fintype Ω] [Nonempty Ω] [Fintype P] [DecidableEq V] (H : Msg → Msg)
    (op : Ω → P → Option V) {vstar : P → V} (hv : IsPlurality op vstar) (c : Ω → ℕ)
    (out : (K : ℕ) → Ω × (Fin K → Ω) → Option (Msg × Msg))
    (hout : ∀ (K : ℕ) (s : Ω × (Fin K → Ω)),
      Found op K s.1 s.2 → ∃ u v, out K s = some (u, v) ∧ Collision H u v)
    (hCR : ∀ K : ℕ, ExpectedCR H (finder op c K (out K))) :
    Pr (fun ω => ∃ p, Off op vstar ω p) ≤
      2 * ((1 + (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ)) * Ex (fun ω : Ω => (c ω : ℝ))) /
        2 ^ (256 : ℝ) := by
  refine le_of_forall_pos_le_add fun ε hε => ?_
  obtain ⟨K, hK⟩ := exists_nat_gt ((#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ) / ε)
  have hK1 : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have h1 := pr_off_le_of_expectedCR_cap H op hv c K (out K) (hout K) (hCR K)
  have h2 : (#(univ.filter fun p : P => (opened op p).Nonempty) : ℝ) / ((K : ℝ) + 1) ≤ ε := by
    rw [div_lt_iff₀ hε] at hK
    rw [div_le_iff₀ hK1]
    nlinarith
  linarith

theorem firstIdx_of_firstHit (op : Ω → P → Option V) (p : P) {w : V} :
    ∀ (K : ℕ) (ρ : Fin K → Ω), firstHit op p K ρ = some w →
      ∃ j, firstIdx op p K ρ = some j ∧ op (ρ j) p = some w := by
  intro K
  induction K with
  | zero =>
    intro ρ h
    simp [firstHit] at h
  | succ K ih =>
    intro ρ h
    cases h0 : op (ρ 0) p with
    | none =>
      have h' : firstHit op p K (fun i => ρ i.succ) = some w := by
        simpa [firstHit, orHit, h0] using h
      obtain ⟨j, hj, hw⟩ := ih (fun i => ρ i.succ) h'
      exact ⟨j.succ, by simp [firstIdx, h0, hj], hw⟩
    | some v =>
      have hvw : v = w := by simpa [firstHit, orHit, h0] using h
      exact ⟨0, by simp [firstIdx, h0], by rw [h0, hvw]⟩

theorem le_loopCost (op : Ω → P → Option V) (c : Ω → ℕ) (p : P) :
    ∀ (K : ℕ) (need : Finset P) (ρ : Fin K → Ω) (j : Fin K), p ∈ need → firstIdx op p K ρ = some j →
      c (ρ j) ≤ loopCost op c K need ρ := by
  intro K
  induction K with
  | zero => intro need ρ j; exact j.elim0
  | succ K ih =>
    intro need ρ j hp hj
    have hn : need.Nonempty := ⟨p, hp⟩
    have e : loopCost op c (K + 1) need ρ =
        c (ρ 0) + loopCost op c K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := by
      simp only [loopCost, hn, ↓reduceIte]
    rw [e]
    cases h0 : op (ρ 0) p with
    | none =>
      have hj' : (firstIdx op p K fun i => ρ i.succ).map Fin.succ = some j := by
        simpa [firstIdx, h0] using hj
      obtain ⟨j', hj1, hj2⟩ := Option.map_eq_some_iff.1 hj'
      have hmem : p ∈ need.filter fun p => (op (ρ 0) p).isNone :=
        Finset.mem_filter.2 ⟨hp, by simp [h0]⟩
      have h := ih (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) j' hmem hj1
      rw [← hj2]
      exact h.trans (Nat.le_add_left _ _)
    | some v =>
      have hj0 : (0 : Fin (K + 1)) = j := by simpa [firstIdx, h0] using hj
      rw [← hj0]
      exact Nat.le_add_right _ _

open Classical in
/-- **The pair the finder outputs**, from `coll`, which reads two runs' tapes: `coll` at the first run and a further
run that is the first to open some node the first run opened, where that is a pair. Every run it reads is one the loop
charges (`le_loopCost`). -/
noncomputable def outOf (op : Ω → P → Option V) (coll : Ω → Ω → Option (Msg × Msg)) (K : ℕ) (ω : Ω)
    (ρ : Fin K → Ω) : Option (Msg × Msg) :=
  if h : ∃ j : Fin K, (∃ p, (op ω p).isSome ∧ firstIdx op p K ρ = some j) ∧ (coll ω (ρ j)).isSome then
    coll ω (ρ h.choose)
  else none

theorem outOf_collision (H : Msg → Msg) (op : Ω → P → Option V) (coll : Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (ω ω' : Ω) (p : P), ConflictAt op (ω, ω') p → (coll ω ω').isSome)
    (hs : ∀ (ω ω' : Ω) (u v : Msg), coll ω ω' = some (u, v) → Collision H u v) {K : ℕ} {ω : Ω} {ρ : Fin K → Ω}
    (h : Found op K ω ρ) : ∃ u v, outOf op coll K ω ρ = some (u, v) ∧ Collision H u v := by
  obtain ⟨p, v, w, hv, hw, hne⟩ := h
  obtain ⟨j, hj, hjw⟩ := firstIdx_of_firstHit op p K ρ hw
  have hex : ∃ j : Fin K, (∃ p, (op ω p).isSome ∧ firstIdx op p K ρ = some j) ∧ (coll ω (ρ j)).isSome :=
    ⟨j, ⟨p, by simp [hv], hj⟩, hc ω (ρ j) p ⟨by simp [hv], by simp [hjw], by simp [hv, hjw, hne]⟩⟩
  obtain ⟨⟨u, v'⟩, huv⟩ := Option.isSome_iff_exists.1 hex.choose_spec.2
  refine ⟨u, v', ?_, hs _ _ u v' huv⟩
  rw [outOf, dite_eq_left hex]
  exact huv

theorem ex_tape {T S : Type} [Fintype T] [Fintype S] (f : T → S → ℝ) :
    Ex (fun q : T × S => f q.1 q.2) = Ex fun t => Ex fun s => f t s := by
  unfold Ex
  rw [Fintype.sum_prod_type' f, Fintype.card_prod, ← Finset.sum_div, Nat.cast_mul, div_div, mul_comm]

theorem pr_tape {T S : Type} [Fintype T] [Fintype S] (p : T → S → Prop) :
    Pr (fun q : T × S => p q.1 q.2) = Ex fun t => Pr fun s => p t s := by
  classical
  have e1 : Pr (fun q : T × S => p q.1 q.2) = Ex fun q : T × S => if p q.1 q.2 then (1 : ℝ) else 0 :=
    Pr_eq_sum _
  have e2 : ∀ t, Pr (fun s => p t s) = Ex fun s => if p t s then (1 : ℝ) else 0 := fun t => Pr_eq_sum _
  rw [e1, ex_tape fun t s => if p t s then (1 : ℝ) else 0]
  exact congrArg Ex (funext fun t => (e2 t).symm)

theorem ex_const_mul {T : Type} [Fintype T] (a : ℝ) (f : T → ℝ) : Ex (fun t => a * f t) = a * Ex f := by
  unfold Ex
  rw [← Finset.mul_sum, mul_div_assoc]

theorem pr_off_tape_le {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [DecidableEq V] (op : T → Ω → P → Option V)
    {vstar : T → P → V} (hv : ∀ t, IsPlurality (op t) (vstar t)) (K : ℕ) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) +
        Pr (fun q : T × (Ω × (Fin K → Ω)) => Miss (op q.1) K q.2.1 q.2.2) := by
  have h : Ex (fun t => Pr fun ω => ∃ p, Off (op t) (vstar t) ω p) ≤
      Ex fun t => 2 * Pr (fun s : Ω × (Fin K → Ω) => Found (op t) K s.1 s.2) +
        Pr (fun s : Ω × (Fin K → Ω) => Miss (op t) K s.1 s.2) :=
    Ex_mono fun t => pr_off_le (op t) (hv t) K
  rw [Ex_add, ex_const_mul] at h
  rw [pr_tape fun t ω => ∃ p, Off (op t) (vstar t) ω p,
    pr_tape fun t (s : Ω × (Fin K → Ω)) => Found (op t) K s.1 s.2,
    pr_tape fun t (s : Ω × (Fin K → Ω)) => Miss (op t) K s.1 s.2]
  exact h

theorem ex_cost_tape_le {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [Fintype P] (op : T → Ω → P → Option V)
    (c : T → Ω → ℕ) (K : ℕ) :
    Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 + loopCost (op q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) ≤
      (1 + (Fintype.card P : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ)) := by
  have h : ∀ t, Ex (fun s : Ω × (Fin K → Ω) =>
      ((c t s.1 + loopCost (op t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (1 + (Fintype.card P : ℝ)) * Ex (fun ω : Ω => (c t ω : ℝ)) := by
    intro t
    have h1 := ex_cost_le (op t) (c t) K
    have h2 : (#(univ.filter fun p : P => (opened (op t) p).Nonempty) : ℝ) ≤ Fintype.card P := by
      exact_mod_cast card_opened_nonempty_le (op t)
    have h3 : (0 : ℝ) ≤ Ex (fun ω : Ω => (c t ω : ℝ)) := Ex_nonneg fun ω => Nat.cast_nonneg _
    nlinarith [mul_le_mul_of_nonneg_right h2 h3]
  calc Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 + loopCost (op q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ))
      = Ex fun t => Ex fun s : Ω × (Fin K → Ω) =>
          ((c t s.1 + loopCost (op t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ) :=
        ex_tape fun t (s : Ω × (Fin K → Ω)) =>
          ((c t s.1 + loopCost (op t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ)
    _ ≤ Ex fun t => (1 + (Fintype.card P : ℝ)) * Ex fun ω : Ω => (c t ω : ℝ) := Ex_mono h
    _ = (1 + (Fintype.card P : ℝ)) * Ex fun t => Ex fun ω : Ω => (c t ω : ℝ) := ex_const_mul _ _
    _ = (1 + (Fintype.card P : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ)) :=
        congrArg (fun z : ℝ => (1 + (Fintype.card P : ℝ)) * z) (ex_tape fun t ω => (c t ω : ℝ)).symm

theorem pr_miss_tape_le {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (K : ℕ) :
    Pr (fun q : T × (Ω × (Fin K → Ω)) => Miss (op q.1) K q.2.1 q.2.2) ≤
      (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have hK : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have h : ∀ t, Pr (fun s : Ω × (Fin K → Ω) => Miss (op t) K s.1 s.2) ≤ (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
    intro t
    have h2 : (#(univ.filter fun p : P => (opened (op t) p).Nonempty) : ℝ) ≤ Fintype.card P := by
      exact_mod_cast card_opened_nonempty_le (op t)
    exact (pr_miss_le (op t) K).trans (div_le_div_of_nonneg_right h2 hK.le)
  calc Pr (fun q : T × (Ω × (Fin K → Ω)) => Miss (op q.1) K q.2.1 q.2.2)
      = Ex fun t => Pr fun s : Ω × (Fin K → Ω) => Miss (op t) K s.1 s.2 :=
        pr_tape fun t (s : Ω × (Fin K → Ω)) => Miss (op t) K s.1 s.2
    _ ≤ Ex fun _ : T => (Fintype.card P : ℝ) / ((K : ℝ) + 1) := Ex_mono h
    _ = (Fintype.card P : ℝ) / ((K : ℝ) + 1) := Ex_const _

/-- **The first-hit finder with the tape in its seed**: its coins are the tape, a first run's coins and `K` further
runs' coins, its output `out`, and its cost the calls of the runs it makes at the tape. -/
noncomputable def tapeFinder {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (c : T → Ω → ℕ) (K : ℕ) (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) :
    Finder where
  S := T × (Ω × (Fin K → Ω))
  out := out
  cost q := c q.1 q.2.1 + loopCost (op q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2

theorem pr_off_tape_le_of_expectedCR {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (c : T → Ω → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg))
    (hout : ∀ q : T × (Ω × (Fin K → Ω)),
      Found (op q.1) K q.2.1 q.2.2 → ∃ u v, out q = some (u, v) ∧ Collision H u v)
    (hCR : ExpectedCR H (tapeFinder op c K out)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * ((1 + (Fintype.card P : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ))) / 2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_tape_le op hv K
  have h2 : Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) ≤
      Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) := Pr_mono hout
  have h3 : Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) ≤
      Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 + loopCost (op q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right (ex_cost_tape_le op c K) (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  have h5 := pr_miss_tape_le op K
  rw [mul_div_assoc]
  linarith

/-- **The further runs' calls, by the sample**: a further tape's, when its sample holds a node of `need` that has not
been opened again. -/
def loopCostOn (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ) : (K : ℕ) → Finset P → (Fin K → Ω) → ℕ
  | 0, _, _ => 0
  | K + 1, need, ρ =>
    (if (need.filter fun p => chk (ρ 0) p).Nonempty then c (ρ 0) else 0) +
      loopCostOn op chk c K (need.filter fun p => (op (ρ 0) p).isNone) fun i => ρ i.succ

theorem loopCostOn_le (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ) (K : ℕ) :
    ∀ (need : Finset P) (ρ : Fin K → Ω),
      loopCostOn op chk c K need ρ ≤ ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c ω else 0) p K ρ := by
  induction K with
  | zero =>
    intro need ρ
    exact Nat.zero_le _
  | succ K ih =>
    intro need ρ
    have e : loopCostOn op chk c (K + 1) need ρ =
        (if (need.filter fun p => chk (ρ 0) p).Nonempty then c (ρ 0) else 0) +
          loopCostOn op chk c K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := rfl
    have e3 : ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c ω else 0) p (K + 1) ρ =
        ∑ p ∈ need, ((if chk (ρ 0) p then c (ρ 0) else 0) +
          if (op (ρ 0) p).isNone then
            nodeCost op (fun ω => if chk ω p then c ω else 0) p K (fun i => ρ i.succ) else 0) :=
      Finset.sum_congr rfl fun _ _ => rfl
    have e2 : ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c ω else 0) p (K + 1) ρ =
        ∑ p ∈ need, (if chk (ρ 0) p then c (ρ 0) else 0) +
          ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone),
            nodeCost op (fun ω => if chk ω p then c ω else 0) p K (fun i => ρ i.succ) := by
      rw [e3, Finset.sum_add_distrib, Finset.sum_filter]
    have h1 : (if (need.filter fun p => chk (ρ 0) p).Nonempty then c (ρ 0) else 0) ≤
        ∑ p ∈ need, (if chk (ρ 0) p then c (ρ 0) else 0) := by
      by_cases hn : (need.filter fun p => chk (ρ 0) p).Nonempty
      · rw [ite_eq_left hn]
        obtain ⟨p, hp⟩ := hn
        obtain ⟨hp1, hp2⟩ := Finset.mem_filter.1 hp
        calc c (ρ 0) = if chk (ρ 0) p then c (ρ 0) else 0 := by simp [hp2]
          _ ≤ ∑ p ∈ need, (if chk (ρ 0) p then c (ρ 0) else 0) :=
            Finset.single_le_sum (f := fun p => if chk (ρ 0) p then c (ρ 0) else 0) (fun _ _ => Nat.zero_le _) hp1
      · rw [ite_eq_right hn]
        exact Nat.zero_le _
    rw [e, e2]
    exact Nat.add_le_add h1 (ih _ _)

theorem le_loopCostOn (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ) (p : P)
    (hchk : ∀ ω, (op ω p).isSome → chk ω p = true) :
    ∀ (K : ℕ) (need : Finset P) (ρ : Fin K → Ω) (j : Fin K), p ∈ need → firstIdx op p K ρ = some j →
      c (ρ j) ≤ loopCostOn op chk c K need ρ := by
  intro K
  induction K with
  | zero => intro need ρ j; exact j.elim0
  | succ K ih =>
    intro need ρ j hp hj
    have e : loopCostOn op chk c (K + 1) need ρ =
        (if (need.filter fun p => chk (ρ 0) p).Nonempty then c (ρ 0) else 0) +
          loopCostOn op chk c K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := rfl
    rw [e]
    cases h0 : op (ρ 0) p with
    | none =>
      have hj' : (firstIdx op p K fun i => ρ i.succ).map Fin.succ = some j := by
        simpa [firstIdx, h0] using hj
      obtain ⟨j', hj1, hj2⟩ := Option.map_eq_some_iff.1 hj'
      have hmem : p ∈ need.filter fun p => (op (ρ 0) p).isNone :=
        Finset.mem_filter.2 ⟨hp, by simp [h0]⟩
      have h := ih (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) j' hmem hj1
      rw [← hj2]
      exact h.trans (Nat.le_add_left _ _)
    | some v =>
      have hj0 : (0 : Fin (K + 1)) = j := by simpa [firstIdx, h0] using hj
      have hne : (need.filter fun p => chk (ρ 0) p).Nonempty :=
        ⟨p, Finset.mem_filter.2 ⟨hp, hchk (ρ 0) (by simp [h0])⟩⟩
      rw [← hj0, ite_eq_left hne]
      exact Nat.le_add_right _ _

theorem sum_costOn_le [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ) (K : ℕ) :
    ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      Fintype.card Ω ^ K * (∑ ω, c ω + ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω) := by
  have h1 : ∀ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      Fintype.card Ω ^ K * c ω +
        ∑ p ∈ univ.filter (fun p => (op ω p).isSome),
          ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c ω else 0) p K ρ := by
    intro ω
    rw [Finset.sum_add_distrib, sum_tapes_const_nat, Finset.sum_comm]
    exact Nat.add_le_add_left (Finset.sum_le_sum fun ρ _ => loopCostOn_le op chk c K _ ρ) _
  have h2 : ∀ p : P, #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c ω else 0) p K ρ ≤
      Fintype.card Ω ^ K * ∑ ω, (if chk ω p then c ω else 0) :=
    fun p => mul_sum_nodeCost_le op (fun ω => if chk ω p then c ω else 0) p K
  have h3 : ∑ p : P, ∑ ω : Ω, (if chk ω p then c ω else 0) = ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
  calc ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ)
      ≤ ∑ ω : Ω, (Fintype.card Ω ^ K * c ω +
          ∑ p ∈ univ.filter (fun p => (op ω p).isSome),
            ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c ω else 0) p K ρ) :=
        Finset.sum_le_sum fun ω _ => h1 ω
    _ = Fintype.card Ω ^ K * ∑ ω, c ω +
          ∑ p, #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c ω else 0) p K ρ := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_need op]
    _ ≤ Fintype.card Ω ^ K * ∑ ω, c ω + ∑ p, Fintype.card Ω ^ K * ∑ ω, (if chk ω p then c ω else 0) :=
        Nat.add_le_add_left (Finset.sum_le_sum fun p _ => h2 p) _
    _ = Fintype.card Ω ^ K * (∑ ω, c ω + ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω) := by
        rw [← Finset.mul_sum, h3, Nat.mul_add]

theorem ex_costOn_le [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ)
    {w : ℕ} (hw : ∀ ω, #(univ.filter fun p : P => chk ω p) ≤ w) (K : ℕ) :
    Ex (fun s : Ω × (Fin K → Ω) =>
        ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (1 + (w : ℝ)) * Ex (fun ω : Ω => (c ω : ℝ)) := by
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hMK : (0 : ℝ) < (Fintype.card Ω : ℝ) ^ K := pow_pos hM K
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow]
  have hsum : ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω ≤ w * ∑ ω, c ω := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun ω _ => Nat.mul_le_mul_right _ (hw ω)
  have hN : ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2) ≤
      (1 + w) * (Fintype.card Ω ^ K * ∑ ω, c ω) :=
    calc ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2)
        = ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ) :=
          Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) =>
            c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ
      _ ≤ Fintype.card Ω ^ K * (∑ ω, c ω + ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω) :=
          sum_costOn_le op chk c K
      _ ≤ Fintype.card Ω ^ K * (∑ ω, c ω + w * ∑ ω, c ω) :=
          Nat.mul_le_mul_left _ (Nat.add_le_add_left hsum _)
      _ = (1 + w) * (Fintype.card Ω ^ K * ∑ ω, c ω) := by ring
  have hR : (∑ s : Ω × (Fin K → Ω),
        ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (1 + (w : ℝ)) * ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ)) := by
    exact_mod_cast hN
  unfold Ex
  rw [hcard]
  calc (∑ s : Ω × (Fin K → Ω),
          ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K)
      ≤ ((1 + (w : ℝ)) * ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ))) /
          ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) :=
        div_le_div_of_nonneg_right hR (mul_pos hM hMK).le
    _ = (1 + (w : ℝ)) * ((∑ ω, (c ω : ℝ)) / (Fintype.card Ω : ℝ)) := by
        rw [show (1 + (w : ℝ)) * ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (c ω : ℝ)) =
            ((1 + (w : ℝ)) * ∑ ω, (c ω : ℝ)) * (Fintype.card Ω : ℝ) ^ K by ring,
          mul_div_mul_right _ _ hMK.ne', mul_div_assoc]

theorem ex_costOn_tape_le {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [Fintype P] (op : T → Ω → P → Option V)
    (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) {w : ℕ} (hw : ∀ t ω, #(univ.filter fun p : P => chk t ω p) ≤ w)
    (K : ℕ) :
    Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) ≤
      (1 + (w : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ)) :=
  calc Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ))
      = Ex fun t => Ex fun s : Ω × (Fin K → Ω) =>
          ((c t s.1 + loopCostOn (op t) (chk t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ) :=
        ex_tape fun t (s : Ω × (Fin K → Ω)) =>
          ((c t s.1 + loopCostOn (op t) (chk t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ)
    _ ≤ Ex fun t => (1 + (w : ℝ)) * Ex fun ω : Ω => (c t ω : ℝ) :=
        Ex_mono fun t => ex_costOn_le (op t) (chk t) (c t) (hw t) K
    _ = (1 + (w : ℝ)) * Ex fun t => Ex fun ω : Ω => (c t ω : ℝ) := ex_const_mul _ _
    _ = (1 + (w : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ)) :=
        congrArg (fun z : ℝ => (1 + (w : ℝ)) * z) (ex_tape fun t ω => (c t ω : ℝ)).symm

/-- **The first-hit finder that runs by the sample**, with the tape in its seed: its coins are the tape, a first
run's coins and `K` further runs' coins, its output `out`, and its cost the first run's calls and each further run's
when that run's sample, at the tape, holds a node still waited for. -/
noncomputable def sampleFinder {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) : Finder where
  S := T × (Ω × (Fin K → Ω))
  out := out
  cost q :=
    c q.1 q.2.1 + loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2

theorem pr_off_sample_le_of_expectedCR {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool) {w : ℕ}
    (hw : ∀ t ω, #(univ.filter fun p : P => chk t ω p) ≤ w) (c : T → Ω → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg))
    (hout : ∀ q : T × (Ω × (Fin K → Ω)),
      Found (op q.1) K q.2.1 q.2.2 → ∃ u v, out q = some (u, v) ∧ Collision H u v)
    (hCR : ExpectedCR H (sampleFinder op chk c K out)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * ((1 + (w : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ))) / 2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_tape_le op hv K
  have h2 : Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) ≤
      Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) := Pr_mono hout
  have h3 : Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) ≤
      Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right (ex_costOn_tape_le op chk c hw K)
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  have h5 := pr_miss_tape_le op K
  rw [mul_div_assoc]
  linarith

/-- `hchk`, that a run opens only nodes of its sample, is a premise the proof doesn't read: it makes the finder's
cost the calls of every run its output reads (`le_loopCostOn`). -/
theorem pr_off_sample_le_of_coll {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (_hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) {w : ℕ}
    (hw : ∀ t ω, #(univ.filter fun p : P => chk t ω p) ≤ w) (c : T → Ω → ℕ) (K : ℕ)
    (coll : T → Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω').isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (u v : Msg), coll t ω ω' = some (u, v) → Collision H u v)
    (hCR : ExpectedCR H
      (sampleFinder op chk c K fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * ((1 + (w : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ))) / 2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) :=
  pr_off_sample_le_of_expectedCR H op hv chk hw c K _
    (fun q h => outOf_collision H (op q.1) (coll q.1) (hc q.1) (hs q.1) h) hCR

theorem le_of_forall_cap {a b n : ℝ} (h : ∀ K : ℕ, a ≤ b + n / ((K : ℝ) + 1)) : a ≤ b := by
  refine le_of_forall_pos_le_add fun ε hε => ?_
  obtain ⟨K, hK⟩ := exists_nat_gt (n / ε)
  have hK1 : (0 : ℝ) < (K : ℝ) + 1 := by positivity
  have h1 := h K
  have h2 : n / ((K : ℝ) + 1) ≤ ε := by
    rw [div_lt_iff₀ hε] at hK
    rw [div_le_iff₀ hK1]
    nlinarith
  linarith

theorem pr_off_sample_le_of_forall_coll {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) {w : ℕ}
    (hw : ∀ t ω, #(univ.filter fun p : P => chk t ω p) ≤ w) (c : T → Ω → ℕ)
    (coll : T → Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω').isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (u v : Msg), coll t ω ω' = some (u, v) → Collision H u v)
    (hCR : ∀ K : ℕ, ExpectedCR H
      (sampleFinder op chk c K fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * ((1 + (w : ℝ)) * Ex (fun q : T × Ω => (c q.1 q.2 : ℝ))) / 2 ^ (256 : ℝ) :=
  le_of_forall_cap fun K => pr_off_sample_le_of_coll H op hv chk hchk hw c K coll hc hs (hCR K)

theorem ex_costOn_le_ex [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ)
    (K : ℕ) :
    Ex (fun s : Ω × (Fin K → Ω) =>
        ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      Ex (fun ω : Ω => (((1 + #(univ.filter fun p : P => chk ω p)) * c ω : ℕ) : ℝ)) := by
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hMK : (0 : ℝ) < (Fintype.card Ω : ℝ) ^ K := pow_pos hM K
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow]
  have hN : ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2) ≤
      Fintype.card Ω ^ K * ∑ ω, (1 + #(univ.filter fun p : P => chk ω p)) * c ω :=
    calc ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2)
        = ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ) :=
          Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) =>
            c ω + loopCostOn op chk c K (univ.filter fun p => (op ω p).isSome) ρ
      _ ≤ Fintype.card Ω ^ K * (∑ ω, c ω + ∑ ω, #(univ.filter fun p : P => chk ω p) * c ω) :=
          sum_costOn_le op chk c K
      _ = Fintype.card Ω ^ K * ∑ ω, (1 + #(univ.filter fun p : P => chk ω p)) * c ω := by
          rw [← Finset.sum_add_distrib]
          exact congrArg (fun z => Fintype.card Ω ^ K * z) (Finset.sum_congr rfl fun ω _ => by ring)
  have hR : (∑ s : Ω × (Fin K → Ω),
        ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (Fintype.card Ω : ℝ) ^ K * ∑ ω, (((1 + #(univ.filter fun p : P => chk ω p)) * c ω : ℕ) : ℝ) := by
    exact_mod_cast hN
  unfold Ex
  rw [hcard]
  calc (∑ s : Ω × (Fin K → Ω),
          ((c s.1 + loopCostOn op chk c K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K)
      ≤ ((Fintype.card Ω : ℝ) ^ K * ∑ ω, (((1 + #(univ.filter fun p : P => chk ω p)) * c ω : ℕ) : ℝ)) /
          ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) :=
        div_le_div_of_nonneg_right hR (mul_pos hM hMK).le
    _ = (∑ ω, (((1 + #(univ.filter fun p : P => chk ω p)) * c ω : ℕ) : ℝ)) / (Fintype.card Ω : ℝ) := by
        rw [mul_comm ((Fintype.card Ω : ℝ) ^ K), mul_div_mul_right _ _ hMK.ne']

theorem ex_costOn_tape_le_ex {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [Fintype P] (op : T → Ω → P → Option V)
    (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (K : ℕ) :
    Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) ≤
      Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) :=
  calc Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ))
      = Ex fun t => Ex fun s : Ω × (Fin K → Ω) =>
          ((c t s.1 + loopCostOn (op t) (chk t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ) :=
        ex_tape fun t (s : Ω × (Fin K → Ω)) =>
          ((c t s.1 + loopCostOn (op t) (chk t) (c t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ)
    _ ≤ Ex fun t => Ex fun ω : Ω => (((1 + #(univ.filter fun p : P => chk t ω p)) * c t ω : ℕ) : ℝ) :=
        Ex_mono fun t => ex_costOn_le_ex (op t) (chk t) (c t) K
    _ = Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) :=
        (ex_tape fun t ω => (((1 + #(univ.filter fun p : P => chk t ω p)) * c t ω : ℕ) : ℝ)).symm

theorem pr_off_sample_le_ex_cap {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg))
    (hout : ∀ q : T × (Ω × (Fin K → Ω)),
      Found (op q.1) K q.2.1 q.2.2 → ∃ u v, out q = some (u, v) ∧ Collision H u v)
    (hCR : ExpectedCR H (sampleFinder op chk c K out)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_tape_le op hv K
  have h2 : Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) ≤
      Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) := Pr_mono hout
  have h3 : Pr (fun q : T × (Ω × (Fin K → Ω)) => ∃ u v, out q = some (u, v) ∧ Collision H u v) ≤
      Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostOn (op q.1) (chk q.1) (c q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right (ex_costOn_tape_le_ex op chk c K)
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  have h5 := pr_miss_tape_le op K
  rw [mul_div_assoc]
  linarith

theorem ex_cost_sampleFinder_le {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) :
    Ex (fun s => ((sampleFinder op chk c K out).cost s : ℝ)) ≤
      Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) :=
  ex_costOn_tape_le_ex op chk c K

theorem pr_off_sample_le_finder {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (K : ℕ)
    (coll : T → Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω').isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (u v : Msg), coll t ω ω' = some (u, v) → Collision H u v) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * (sampleFinder op chk c K
          fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2).Pr H +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_tape_le op hv K
  have h2 : Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) ≤
      Pr (fun q : T × (Ω × (Fin K → Ω)) =>
        ∃ u v, outOf (op q.1) (coll q.1) K q.2.1 q.2.2 = some (u, v) ∧ Collision H u v) :=
    Pr_mono fun q h => outOf_collision H (op q.1) (coll q.1) (hc q.1) (hs q.1) h
  have h3 : Pr (fun q : T × (Ω × (Fin K → Ω)) =>
        ∃ u v, outOf (op q.1) (coll q.1) K q.2.1 q.2.2 = some (u, v) ∧ Collision H u v) =
      (sampleFinder op chk c K
        fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2).Pr H := rfl
  have h5 := pr_miss_tape_le op K
  rw [← h3]
  linarith

/-- `hchk`, that a run opens only nodes of its sample, is a premise the proof doesn't read: it makes the cost the
calls of every run the output reads (`le_loopCostOn`). -/
theorem pr_off_sample_le_ex_cap_of_coll {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (_hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) (c : T → Ω → ℕ) (K : ℕ)
    (coll : T → Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω').isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (u v : Msg), coll t ω ω' = some (u, v) → Collision H u v)
    (hCR : ExpectedCR H
      (sampleFinder op chk c K fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) :=
  pr_off_sample_le_ex_cap H op hv chk c K _
    (fun q h => outOf_collision H (op q.1) (coll q.1) (hc q.1) (hs q.1) h) hCR

theorem pr_off_sample_le_ex {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) (c : T → Ω → ℕ)
    (coll : T → Ω → Ω → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω').isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (u v : Msg), coll t ω ω' = some (u, v) → Collision H u v)
    (hCR : ∀ K : ℕ, ExpectedCR H
      (sampleFinder op chk c K fun q : T × (Ω × (Fin K → Ω)) => outOf (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Ex (fun q : T × Ω => (((1 + #(univ.filter fun p : P => chk q.1 q.2 p)) * c q.1 q.2 : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  le_of_forall_cap fun K => pr_off_sample_le_ex_cap_of_coll H op hv chk hchk c K coll hc hs (hCR K)

theorem loopCostAt_le (op : Ω → P → Option V) (chk : Ω → P → Bool) (c₂ : Ω → Finset P → ℕ) (c₁ : Ω → P → ℕ)
    (hsub : ∀ ω J, c₂ ω J ≤ ∑ p ∈ J, c₁ ω p) (K : ℕ) :
    ∀ (need : Finset P) (ρ : Fin K → Ω),
      loopCostAt op chk c₂ K need ρ ≤ ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K ρ := by
  induction K with
  | zero =>
    intro need ρ
    exact Nat.zero_le _
  | succ K ih =>
    intro need ρ
    have e : loopCostAt op chk c₂ (K + 1) need ρ =
        c₂ (ρ 0) (need.filter fun p => chk (ρ 0) p) +
          loopCostAt op chk c₂ K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := rfl
    have e3 : ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p (K + 1) ρ =
        ∑ p ∈ need, ((if chk (ρ 0) p then c₁ (ρ 0) p else 0) +
          if (op (ρ 0) p).isNone then
            nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K (fun i => ρ i.succ) else 0) :=
      Finset.sum_congr rfl fun _ _ => rfl
    have e2 : ∑ p ∈ need, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p (K + 1) ρ =
        ∑ p ∈ need, (if chk (ρ 0) p then c₁ (ρ 0) p else 0) +
          ∑ p ∈ need.filter (fun p => (op (ρ 0) p).isNone),
            nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K (fun i => ρ i.succ) := by
      rw [e3, Finset.sum_add_distrib, Finset.sum_filter]
    have h1 : c₂ (ρ 0) (need.filter fun p => chk (ρ 0) p) ≤
        ∑ p ∈ need, (if chk (ρ 0) p then c₁ (ρ 0) p else 0) :=
      (hsub _ _).trans (le_of_eq (Finset.sum_filter _ _))
    rw [e, e2]
    exact Nat.add_le_add h1 (ih _ _)

theorem le_loopCostAt (op : Ω → P → Option V) (chk : Ω → P → Bool) (c₂ : Ω → Finset P → ℕ) (p : P) (a : Ω → ℕ)
    (hchk : ∀ ω, (op ω p).isSome → chk ω p = true) (ha : ∀ ω J, p ∈ J → a ω ≤ c₂ ω J) :
    ∀ (K : ℕ) (need : Finset P) (ρ : Fin K → Ω) (j : Fin K), p ∈ need → firstIdx op p K ρ = some j →
      a (ρ j) ≤ loopCostAt op chk c₂ K need ρ := by
  intro K
  induction K with
  | zero => intro need ρ j; exact j.elim0
  | succ K ih =>
    intro need ρ j hp hj
    have e : loopCostAt op chk c₂ (K + 1) need ρ =
        c₂ (ρ 0) (need.filter fun p => chk (ρ 0) p) +
          loopCostAt op chk c₂ K (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := rfl
    rw [e]
    cases h0 : op (ρ 0) p with
    | none =>
      have hj' : (firstIdx op p K fun i => ρ i.succ).map Fin.succ = some j := by
        simpa [firstIdx, h0] using hj
      obtain ⟨j', hj1, hj2⟩ := Option.map_eq_some_iff.1 hj'
      have hmem : p ∈ need.filter fun p => (op (ρ 0) p).isNone :=
        Finset.mem_filter.2 ⟨hp, by simp [h0]⟩
      have h := ih (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) j' hmem hj1
      rw [← hj2]
      exact h.trans (Nat.le_add_left _ _)
    | some v =>
      have hj0 : (0 : Fin (K + 1)) = j := by simpa [firstIdx, h0] using hj
      have hmem : p ∈ need.filter fun p => chk (ρ 0) p :=
        Finset.mem_filter.2 ⟨hp, hchk (ρ 0) (by simp [h0])⟩
      rw [← hj0]
      exact (ha (ρ 0) _ hmem).trans (Nat.le_add_right _ _)

theorem sum_costAt_le [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (chk : Ω → P → Bool) (c : Ω → ℕ)
    (c₂ : Ω → Finset P → ℕ) (c₁ : Ω → P → ℕ) (hsub : ∀ ω J, c₂ ω J ≤ ∑ p ∈ J, c₁ ω p) (K : ℕ) :
    ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostAt op chk c₂ K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      Fintype.card Ω ^ K * ∑ ω, (c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p) := by
  have h1 : ∀ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostAt op chk c₂ K (univ.filter fun p => (op ω p).isSome) ρ) ≤
      Fintype.card Ω ^ K * c ω +
        ∑ p ∈ univ.filter (fun p => (op ω p).isSome),
          ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K ρ := by
    intro ω
    rw [Finset.sum_add_distrib, sum_tapes_const_nat, Finset.sum_comm]
    exact Nat.add_le_add_left (Finset.sum_le_sum fun ρ _ => loopCostAt_le op chk c₂ c₁ hsub K _ ρ) _
  have h2 : ∀ p : P,
      #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K ρ ≤
        Fintype.card Ω ^ K * ∑ ω, (if chk ω p then c₁ ω p else 0) :=
    fun p => mul_sum_nodeCost_le op (fun ω => if chk ω p then c₁ ω p else 0) p K
  have h3 : ∑ p : P, ∑ ω : Ω, (if chk ω p then c₁ ω p else 0) =
      ∑ ω, ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun ω _ => (Finset.sum_filter _ _).symm
  calc ∑ ω : Ω, ∑ ρ : Fin K → Ω, (c ω + loopCostAt op chk c₂ K (univ.filter fun p => (op ω p).isSome) ρ)
      ≤ ∑ ω : Ω, (Fintype.card Ω ^ K * c ω +
          ∑ p ∈ univ.filter (fun p => (op ω p).isSome),
            ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K ρ) :=
        Finset.sum_le_sum fun ω _ => h1 ω
    _ = Fintype.card Ω ^ K * ∑ ω, c ω +
          ∑ p, #(opened op p) * ∑ ρ : Fin K → Ω, nodeCost op (fun ω => if chk ω p then c₁ ω p else 0) p K ρ := by
        rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_need op]
    _ ≤ Fintype.card Ω ^ K * ∑ ω, c ω + ∑ p, Fintype.card Ω ^ K * ∑ ω, (if chk ω p then c₁ ω p else 0) :=
        Nat.add_le_add_left (Finset.sum_le_sum fun p _ => h2 p) _
    _ = Fintype.card Ω ^ K * ∑ ω, (c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p) := by
        rw [← Finset.mul_sum, h3, ← Nat.mul_add, ← Finset.sum_add_distrib]

theorem ex_costAt_le_ex [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (chk : Ω → P → Bool)
    (c : Ω → ℕ) (c₂ : Ω → Finset P → ℕ) (c₁ : Ω → P → ℕ) (hsub : ∀ ω J, c₂ ω J ≤ ∑ p ∈ J, c₁ ω p) (K : ℕ) :
    Ex (fun s : Ω × (Fin K → Ω) =>
        ((c s.1 + loopCostAt op chk c₂ K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      Ex (fun ω : Ω => ((c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p : ℕ) : ℝ)) := by
  have hM : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast Fintype.card_pos
  have hMK : (0 : ℝ) < (Fintype.card Ω : ℝ) ^ K := pow_pos hM K
  have hcard : (Fintype.card (Ω × (Fin K → Ω)) : ℝ) = (Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K := by
    rw [Fintype.card_prod, Fintype.card_pi_const, Nat.cast_mul, Nat.cast_pow]
  have hN : ∑ s : Ω × (Fin K → Ω), (c s.1 + loopCostAt op chk c₂ K (univ.filter fun p => (op s.1 p).isSome) s.2) ≤
      Fintype.card Ω ^ K * ∑ ω, (c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p) :=
    (Fintype.sum_prod_type' fun (ω : Ω) (ρ : Fin K → Ω) =>
      c ω + loopCostAt op chk c₂ K (univ.filter fun p => (op ω p).isSome) ρ).trans_le
      (sum_costAt_le op chk c c₂ c₁ hsub K)
  have hR : (∑ s : Ω × (Fin K → Ω),
        ((c s.1 + loopCostAt op chk c₂ K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) ≤
      (Fintype.card Ω : ℝ) ^ K *
        ∑ ω, ((c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p : ℕ) : ℝ) := by
    exact_mod_cast hN
  unfold Ex
  rw [hcard]
  calc (∑ s : Ω × (Fin K → Ω),
          ((c s.1 + loopCostAt op chk c₂ K (univ.filter fun p => (op s.1 p).isSome) s.2 : ℕ) : ℝ)) /
        ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K)
      ≤ ((Fintype.card Ω : ℝ) ^ K *
            ∑ ω, ((c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p : ℕ) : ℝ)) /
          ((Fintype.card Ω : ℝ) * (Fintype.card Ω : ℝ) ^ K) :=
        div_le_div_of_nonneg_right hR (mul_pos hM hMK).le
    _ = (∑ ω, ((c ω + ∑ p ∈ univ.filter (fun p : P => chk ω p), c₁ ω p : ℕ) : ℝ)) / (Fintype.card Ω : ℝ) := by
        rw [mul_comm ((Fintype.card Ω : ℝ) ^ K), mul_div_mul_right _ _ hMK.ne']

theorem ex_costAt_tape_le_ex {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (c₂ : T → Ω → Finset P → ℕ)
    (c₁ : T → Ω → P → ℕ) (hsub : ∀ t ω J, c₂ t ω J ≤ ∑ p ∈ J, c₁ t ω p) (K : ℕ) :
    Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostAt (op q.1) (chk q.1) (c₂ q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ)) ≤
      Ex (fun q : T × Ω =>
        ((c q.1 q.2 + ∑ p ∈ univ.filter (fun p : P => chk q.1 q.2 p), c₁ q.1 q.2 p : ℕ) : ℝ)) :=
  calc Ex (fun q : T × (Ω × (Fin K → Ω)) =>
        ((c q.1 q.2.1 +
          loopCostAt (op q.1) (chk q.1) (c₂ q.1) K (univ.filter fun p => (op q.1 q.2.1 p).isSome) q.2.2 : ℕ) : ℝ))
      = Ex fun t => Ex fun s : Ω × (Fin K → Ω) =>
          ((c t s.1 + loopCostAt (op t) (chk t) (c₂ t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ) :=
        ex_tape fun t (s : Ω × (Fin K → Ω)) =>
          ((c t s.1 + loopCostAt (op t) (chk t) (c₂ t) K (univ.filter fun p => (op t s.1 p).isSome) s.2 : ℕ) : ℝ)
    _ ≤ Ex fun t => Ex fun ω : Ω =>
          ((c t ω + ∑ p ∈ univ.filter (fun p : P => chk t ω p), c₁ t ω p : ℕ) : ℝ) :=
        Ex_mono fun t => ex_costAt_le_ex (op t) (chk t) (c t) (c₂ t) (c₁ t) (hsub t) K
    _ = Ex (fun q : T × Ω =>
          ((c q.1 q.2 + ∑ p ∈ univ.filter (fun p : P => chk q.1 q.2 p), c₁ q.1 q.2 p : ℕ) : ℝ)) :=
        (ex_tape fun t ω => ((c t ω + ∑ p ∈ univ.filter (fun p : P => chk t ω p), c₁ t ω p : ℕ) : ℝ)).symm

theorem ex_cost_sampleFinderAt_le {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (c₂ : T → Ω → Finset P → ℕ)
    (c₁ : T → Ω → P → ℕ) (hsub : ∀ t ω J, c₂ t ω J ≤ ∑ p ∈ J, c₁ t ω p) (K : ℕ)
    (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) :
    Ex (fun s => ((sampleFinderAt op chk c c₂ K out).cost s : ℝ)) ≤
      Ex (fun q : T × Ω =>
        ((c q.1 q.2 + ∑ p ∈ univ.filter (fun p : P => chk q.1 q.2 p), c₁ q.1 q.2 p : ℕ) : ℝ)) :=
  ex_costAt_tape_le_ex op chk c c₂ c₁ hsub K

theorem outOfAt_collision (H : Msg → Msg) (op : Ω → P → Option V) (coll : Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (ω ω' : Ω) (p : P), ConflictAt op (ω, ω') p → (coll ω ω' p).isSome)
    (hs : ∀ (ω ω' : Ω) (p : P) (u v : Msg), coll ω ω' p = some (u, v) → Collision H u v) {K : ℕ} {ω : Ω}
    {ρ : Fin K → Ω} (h : Found op K ω ρ) : ∃ u v, outOfAt op coll K ω ρ = some (u, v) ∧ Collision H u v := by
  obtain ⟨p, v, w, hv, hw, hne⟩ := h
  obtain ⟨j, hj, hjw⟩ := firstIdx_of_firstHit op p K ρ hw
  have hex : ∃ jp : Fin K × P, ((op ω jp.2).isSome ∧ firstIdx op jp.2 K ρ = some jp.1) ∧
      (coll ω (ρ jp.1) jp.2).isSome :=
    ⟨(j, p), ⟨by simp [hv], hj⟩, hc ω (ρ j) p ⟨by simp [hv], by simp [hjw], by simp [hv, hjw, hne]⟩⟩
  obtain ⟨⟨u, v'⟩, huv⟩ := Option.isSome_iff_exists.1 hex.choose_spec.2
  refine ⟨u, v', ?_, hs _ _ _ u v' huv⟩
  rw [outOfAt, dite_eq_left hex]
  exact huv

theorem pr_off_at_le_finder {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ)
    (c₂ : T → Ω → Finset P → ℕ) (K : ℕ) (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * (sampleFinderAt op chk c c₂ K
          fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2).Pr H +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_tape_le op hv K
  have h2 : Pr (fun q : T × (Ω × (Fin K → Ω)) => Found (op q.1) K q.2.1 q.2.2) ≤
      Pr (fun q : T × (Ω × (Fin K → Ω)) =>
        ∃ u v, outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2 = some (u, v) ∧ Collision H u v) :=
    Pr_mono fun q h => outOfAt_collision H (op q.1) (coll q.1) (hc q.1) (hs q.1) h
  have h3 : Pr (fun q : T × (Ω × (Fin K → Ω)) =>
        ∃ u v, outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2 = some (u, v) ∧ Collision H u v) =
      (sampleFinderAt op chk c c₂ K
        fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2).Pr H := rfl
  have h5 := pr_miss_tape_le op K
  rw [← h3]
  linarith

/-- `hchk`, that a run opens only nodes of its sample, is a premise the proof doesn't read: under it the loop
charges what the output reads (`le_loopCostAt`). -/
theorem pr_off_at_le_ex_cap {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (_hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) (c : T → Ω → ℕ)
    (c₂ : T → Ω → Finset P → ℕ) (c₁ : T → Ω → P → ℕ) (hsub : ∀ t ω J, c₂ t ω J ≤ ∑ p ∈ J, c₁ t ω p) (K : ℕ)
    (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v)
    (hCR : ExpectedCR H (sampleFinderAt op chk c c₂ K
      fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Ex (fun q : T × Ω =>
          ((c q.1 q.2 + ∑ p ∈ univ.filter (fun p : P => chk q.1 q.2 p), c₁ q.1 q.2 p : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_at_le_finder H op hv chk c c₂ K coll hc hs
  have h3 : (sampleFinderAt op chk c c₂ K
        fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2).Pr H ≤
      Ex (fun s => ((sampleFinderAt op chk c c₂ K
        fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2).cost s : ℝ)) /
        2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right
    (ex_cost_sampleFinderAt_le op chk c c₂ c₁ hsub K
      fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2)
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  rw [mul_div_assoc]
  linarith

theorem pr_off_at_le_ex {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq V] (H : Msg → Msg) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true) (c : T → Ω → ℕ)
    (c₂ : T → Ω → Finset P → ℕ) (c₁ : T → Ω → P → ℕ) (hsub : ∀ t ω J, c₂ t ω J ≤ ∑ p ∈ J, c₁ t ω p)
    (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v)
    (hCR : ∀ K : ℕ, ExpectedCR H (sampleFinderAt op chk c c₂ K
      fun q : T × (Ω × (Fin K → Ω)) => outOfAt (op q.1) (coll q.1) K q.2.1 q.2.2)) :
    Pr (fun q : T × Ω => ∃ p, Off (op q.1) (vstar q.1) q.2 p) ≤
      2 * Ex (fun q : T × Ω =>
          ((c q.1 q.2 + ∑ p ∈ univ.filter (fun p : P => chk q.1 q.2 p), c₁ q.1 q.2 p : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  le_of_forall_cap fun K => pr_off_at_le_ex_cap H op hv chk hchk c c₂ c₁ hsub K coll hc hs (hCR K)

end Verity.FirstHit
