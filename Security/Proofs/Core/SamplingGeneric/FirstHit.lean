module

public import Definitions.Core.Interaction
public import Definitions.Core.Rewinding
public import Proofs.Flock.Claims.Prob

/-!
# The first-hit finder: openings off the plurality table cost collisions, linearly

The rewinding lemma (`Proofs.Core.Game.Rewinding`) bounds the runs that open some position off the plurality table by
the square root of the conflicting pairs of runs. This file bounds them linearly, by one collision finder's success, in
the same vocabulary (`Definitions.Core.Rewinding`): a run with tape `ω` opens node `p` to `op ω p`, and `vstar` is a
plurality table (`IsPlurality`).

**The finder.** Its seed is a first tape `ω` and `K` further tapes `ρ 0, …, ρ (K − 1)`, all uniform and independent
(`Ω × (Fin K → Ω)`). It runs `ω`, then the further tapes in order until every node the first run opened has been opened
again or the `K` tapes are used up. At each node the first run opened, it compares the first run's value with the value
of the first further tape that opens the node (`firstHit`), and it finds a pair when the two differ at some node
(`Found`). Its cap runs out when no further tape opens some node the first run opened (`Miss`).

* **Success** (`pr_off_le`): `Pr[a run opens a node off the plurality] ≤ 2·Pr[Found] + Pr[Miss]`. At a node the first
  run opens to a value `v` off the plurality, at most half the tapes that open the node open it to `v`
  (`two_mul_openedTo_le`), so the first further tape that opens it differs from `v` at least half the time
  (`pow_le_sum_score`). The bound is linear in the event, so it averages over anything outside with no square root.
* **Cost** (`ex_cost_le`): with `c ω` the calls of a run on tape `ω`, the finder's expected calls are at most
  `(1 + w)·E[c]`, where `w` is the number of nodes that some tape opens: a count over the node type, at most its
  size, and neither the most nodes one run opens nor their expected number. Waiting for a node that a run opens with
  probability `q` takes at most `1/q` further runs in expectation, and the first run opens it with probability `q`, so
  each node some tape opens adds at most one run (`mul_sum_nodeCost_le`); a further tape's calls are charged only while
  some node is still waited for, which the earlier tapes decide (`loopCost`, `loopCost_le`).
* **Under `ExpectedCR`** (`pr_off_le_of_expectedCR`): when every found pair is a collision of `H` that the finder
  outputs, and the finder (`finder`, its cost the calls of the runs it makes) meets the linear expected-time bound,
  `Pr[a run opens a node off the plurality] ≤ 2·(1 + w)·E[c]/2^256 + Pr[Miss]`.

The cap's term `Pr[Miss]` is stated as the event's probability.
-/

@[expose] public section

open Finset

namespace Verity.FirstHit

open Verity.Rewinding Claims

/-! ## Sums over sequences of tapes -/

/-- A sum over `K + 1` tapes is a sum over the first tape and the `K` others. -/
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

/-! ## The finder -/

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

/-! ## Success: at least half the off-plurality mass -/

/-- **Plurality**: of the tapes that open `p`, at most half open it to a value `v` other than the plurality's. -/
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

/-- One further tape: the scores against `v` sum to at least the number of tapes, when at most half the tapes that open
`p` open it to `v`. -/
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
      · simp [score, hw] <;> norm_num
      · simp [score, hw] <;> norm_num
  have e : ∑ ω, (score v (op ω p) + 2 * (if op ω p = some v then (1 : ℝ) else 0)) =
      ∑ ω : Ω, ((1 : ℝ) + (if (op ω p).isSome then (1 : ℝ) else 0)) :=
    Finset.sum_congr rfl fun ω _ => hpt ω
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ,
    nsmul_eq_mul, mul_one, ← h1, ← h2] at e
  linarith

/-- `K` further tapes: the first hits' scores against `v` sum to at least the number of sequences. -/
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

/-- A node the first run opens to `v` scores at most twice a found pair plus a miss. -/
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
    rw [if_pos hm, h1]
    linarith
  | some w =>
    by_cases hw : w = v
    · have h0 : score v (some w) = 0 := by simp [score, hw]
      rw [h0]
      linarith
    · have hfound : Found op K ω ρ := Exists.intro p (Exists.intro v (Exists.intro w ⟨h, hf, fun e => hw e.symm⟩))
      have h2 : score v (some w) = 2 := by simp [score, hw]
      rw [if_pos hfound, h2]
      linarith

/-- At one first tape: an opening off the plurality, times the sequences of further tapes, is at most twice the
sequences that find a pair plus those that miss. -/
theorem off_mul_pow_le [Fintype Ω] [DecidableEq V] (op : Ω → P → Option V) {vstar : P → V}
    (hv : IsPlurality op vstar) (K : ℕ) (ω : Ω) [Decidable (∃ p, Off op vstar ω p)]
    [DecidablePred (Found op K ω)] [DecidablePred (Miss op K ω)] :
    (if ∃ p, Off op vstar ω p then (1 : ℝ) else 0) * (Fintype.card Ω : ℝ) ^ K ≤
      ∑ ρ : Fin K → Ω,
        (2 * (if Found op K ω ρ then (1 : ℝ) else 0) + (if Miss op K ω ρ then (1 : ℝ) else 0)) := by
  by_cases hb : ∃ p, Off op vstar ω p
  · rw [if_pos hb, one_mul]
    obtain ⟨p, hs, hne⟩ := hb
    obtain ⟨v, hv'⟩ := Option.isSome_iff_exists.1 hs
    have hvne : v ≠ vstar p := fun e => hne (by rw [hv', e])
    exact (pow_le_sum_score op (two_mul_openedTo_le op hv hvne) K).trans
      (Finset.sum_le_sum fun ρ _ => score_le op K hv' ρ)
  · rw [if_neg hb, zero_mul]
    exact Finset.sum_nonneg fun ρ _ =>
      add_nonneg (mul_nonneg zero_le_two (indicator_nonneg _)) (indicator_nonneg _)

/-- **Success.** A uniform run opens some node off the plurality table with probability at most twice the finder's
probability of finding a pair, plus the probability that its cap runs out. -/
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

/-! ## Cost: one first run, and at most one further run for each node some tape opens -/

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

/-- The further runs cost at most what waiting for each node of `need` on its own costs. -/
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

/-- The tapes that open `p` and those that don't are all the tapes. -/
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

/-- One more tape in front: its calls, and the wait's when it doesn't open `p`. -/
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

/-- **Waiting for one node**: the tapes that open `p`, times the wait's calls summed over the sequences, are at most the
sequences times one run's calls summed over the tapes. In expectation: a node opened with probability `q` is waited for
at a cost of at most `E[c]/q`. -/
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

/-- Summing a node's weight over the nodes each tape opens counts it once for each tape that opens it. -/
theorem sum_need [Fintype Ω] [Fintype P] (op : Ω → P → Option V) (g : P → ℕ) :
    ∑ ω : Ω, ∑ p ∈ univ.filter (fun p => (op ω p).isSome), g p = ∑ p, #(opened op p) * g p :=
  calc ∑ ω : Ω, ∑ p ∈ univ.filter (fun p => (op ω p).isSome), g p
      = ∑ ω : Ω, ∑ p, if (op ω p).isSome then g p else 0 :=
        Finset.sum_congr rfl fun ω _ => Finset.sum_filter _ _
    _ = ∑ p, ∑ ω : Ω, if (op ω p).isSome then g p else 0 := Finset.sum_comm
    _ = ∑ p, #(opened op p) * g p := Finset.sum_congr rfl fun p _ =>
        (by rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul] :
          (∑ ω : Ω, if (op ω p).isSome then g p else 0) = #(univ.filter fun ω : Ω => (op ω p).isSome) * g p)

/-- **The finder's calls, summed over its seeds**: at most `1 + w` times the sequences times one run's calls summed
over the tapes, `w` the number of nodes some tape opens. -/
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
    · rw [if_pos hp]
      exact mul_sum_nodeCost_le op c p K
    · have h0 : #(opened op p) = 0 := Finset.card_eq_zero.2 (Finset.not_nonempty_iff_eq_empty.1 hp)
      exact le_of_eq (by rw [if_neg hp, h0, Nat.zero_mul])
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

/-- **Cost.** The finder's expected calls are at most `1 + w` times a run's expected calls, where `w` is the number of
nodes that some tape opens (at most the number of nodes): the first run, and for each such node at most one further run
in expectation. -/
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

/-! ## Under the linear expected-time bound -/

/-- **The first-hit finder** as a collision finder: its coins are a first tape and `K` further tapes, its output `out`,
and its cost the calls of the runs it makes: the first tape's, and each further tape's while some node the first run
opened has not been opened again. -/
noncomputable def finder [Fintype Ω] [Nonempty Ω] [Fintype P] (op : Ω → P → Option V) (c : Ω → ℕ) (K : ℕ)
    (out : Ω × (Fin K → Ω) → Option (Msg × Msg)) : Finder where
  S := Ω × (Fin K → Ω)
  out := out
  cost s := c s.1 + loopCost op c K (univ.filter fun p => (op s.1 p).isSome) s.2

/-- **The off-plurality mass under `ExpectedCR`.** If every pair the finder finds is a collision of `H` that it
outputs, and the finder meets the linear expected-time bound, a uniform run opens some node off the plurality table with
probability at most `2·(1 + w)·E[c]/2^256`, plus the probability that the finder's cap runs out; `c` counts a run's
calls and `w` the nodes some tape opens. -/
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

end Verity.FirstHit
