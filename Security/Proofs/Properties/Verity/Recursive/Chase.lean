import Proofs.Properties.Verity.Recursive.Target

/-!
# The target at the chasing finder, and the target

`sampledProofsRecursive_chase` proves the target's bound at each strategy whose finders meet `ChaseCR`, `Dep.deltaChase`,
and `sampledProofsRecursive` the smaller of it and `sampledProofsRecursive_bind`'s (`Dep.errAt`). Both start from
`prob_le_off`, which leaves the probability that a play is off the plurality (`offEvent`).

The chasing finder (`Dep.chaseAt`) plays once; where the play is off the plurality at the least unit, wire and bit `b`
it reads off (`Dep.chaseSel`), it reads `M_b = ⌈β / q_b⌉` further plays at `b`, with `q_b` the probability that a play
reads `b` (`Dep.weight` at both values) and `β = √(8/n)`, `n` the units, and hands `collOf` the first play and the first
further one that reads the plurality's bit there (`Dep.chasePair`):
* the plurality's bit weighs at least the off bit (`minority_le_majority`), so at least `q_b / 2`, and `M_b` plays all
  miss it with probability at most `e^{−β/2}`: the finder holds such a pair with probability at least
  `(1 − e^{−β/2})` times the probability that a play is off (`chase_success`), and `collOf` turns the pair into a
  collision (`hcoll`, `chase_coll`);
* a pick at the unit `u` and the bit `b` draws `u`, which weighs at most `q_b` (`drawn_le_sum`), so the finder reads at
  most `2 + β n` plays in expectation (`chase_cost`), and with each play's calls at most its budget
  (`Dep.ex_playCost_le`) and `collOf`'s at most `dColl`, its expected calls are at most `Dep.chaseBudget`
  (`ex_chaseAt_cost_le`);
* so the probability that a play is off is at most `chaseBudget / ((1 − e^{−β/2}) · 2^256)` under `ecr/sha-512` at the
  finder (`ex_off_le_chase`).
-/

namespace Verity.Recursive

open scoped ENNReal
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy

namespace Target

open Claims

/-! ## Uniform tables -/

/-- An average of one entry of a uniform table is that entry's average. -/
theorem ex_pi_eval {Ξ : Type} [Fintype Ξ] [DecidableEq Ξ] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    [∀ ξ, Nonempty (Smp ξ)] (ξ : Ξ) (f : Smp ξ → ℝ) :
    Ex (fun T : (ξ' : Ξ) → Smp ξ' => f (T ξ)) = Ex f := by
  unfold Ex
  have hsum : ∑ T : (ξ' : Ξ) → Smp ξ', f (T ξ) =
      ∑ p : Smp ξ × ((j : { j // j ≠ ξ }) → Smp j), f p.1 :=
    Fintype.sum_equiv (Equiv.piSplitAt ξ Smp) _ _ fun T => rfl
  have h1 : (0 : ℝ) < Fintype.card ((j : { j // j ≠ ξ }) → Smp j) := by exact_mod_cast Fintype.card_pos
  have h2 : (0 : ℝ) < Fintype.card (Smp ξ) := by exact_mod_cast Fintype.card_pos
  rw [hsum, Fintype.sum_prod_type, Fintype.card_congr (Equiv.piSplitAt ξ Smp), Fintype.card_prod]
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [← Finset.mul_sum]
  push_cast
  field_simp

theorem pr_pi_eval {Ξ : Type} [Fintype Ξ] [DecidableEq Ξ] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    [∀ ξ, Nonempty (Smp ξ)] (ξ : Ξ) (P : Smp ξ → Prop) :
    Pr (fun T : (ξ' : Ξ) → Smp ξ' => P (T ξ)) = Pr P := by
  classical
  rw [← ex_ind, ← ex_ind]
  exact ex_pi_eval ξ (fun s => if P s then 1 else 0)

/-- **A uniform play**: a uniform draw and a uniform table of samples, read at the draw, is a uniform draw and a
uniform sample at it. -/
theorem ex_table {Ξ : Type} [Fintype Ξ] [DecidableEq Ξ] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    [∀ ξ, Nonempty (Smp ξ)] (f : (ξ : Ξ) → Smp ξ → ℝ) :
    Ex (fun p : Ξ × ((ξ : Ξ) → Smp ξ) => f p.1 (p.2 p.1)) = Ex (fun ξ => Ex (f ξ)) :=
  (ex_prod (fun ξ (T : (ξ' : Ξ) → Smp ξ') => f ξ (T ξ))).trans (ex_congr fun ξ => ex_pi_eval ξ (f ξ))

theorem pr_table {Ξ : Type} [Fintype Ξ] [DecidableEq Ξ] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    [∀ ξ, Nonempty (Smp ξ)] (P : (ξ : Ξ) → Smp ξ → Prop) :
    Pr (fun p : Ξ × ((ξ : Ξ) → Smp ξ) => P p.1 (p.2 p.1)) = Ex (fun ξ => Pr (P ξ)) := by
  classical
  rw [← ex_ind]
  refine (ex_table (fun ξ s => if P ξ s then (1 : ℝ) else 0)).trans (ex_congr fun ξ => ex_ind (P ξ))

theorem ex_add_const {β : Type} [Fintype β] [Nonempty β] (f : β → ℝ) (c : ℝ) :
    Ex (fun a => f a + c) = Ex f + c := by
  rw [Ex_add f (fun _ => c), Ex_const]

theorem pr_not {β : Type} [Fintype β] [Nonempty β] (p : β → Prop) : Pr (fun a => ¬ p a) = 1 - Pr p := by
  classical
  rw [eq_sub_iff_add_eq, ← ex_ind, ← ex_ind, ← Ex_add]
  rw [ex_congr (g := fun _ => (1 : ℝ)) fun a => by by_cases h : p a <;> simp [h]]
  exact Ex_const 1

theorem pr_exists_pi {X : Type} [Fintype X] [Nonempty X] (m : ℕ) (A : X → Prop) :
    Pr (fun y : Fin m → X => ∃ j, A (y j)) = 1 - (1 - Pr A) ^ m := by
  have h1 : Pr (fun y : Fin m → X => ∃ j, A (y j)) = 1 - Pr (fun y : Fin m → X => ∀ j, ¬ A (y j)) := by
    rw [← pr_not]
    exact Pr_congr fun y => by simp only [not_forall, not_not]
  rw [h1, Pr_pi m (fun x => ¬ A x), pr_not]

theorem one_sub_pow_le_exp {w : ℝ} (hw1 : w ≤ 1) (M : ℕ) : (1 - w) ^ M ≤ Real.exp (-(w * M)) := by
  calc (1 - w) ^ M ≤ Real.exp (-w) ^ M :=
        pow_le_pow_left₀ (by linarith) (by linarith [Real.add_one_le_exp (-w)]) M
    _ = Real.exp (-(w * M)) := by rw [← Real.exp_nat_mul]; congr 1; ring

/-- **`m` independent tries hit an event of probability `a`** with probability at least `1 − e^{−a m}`. -/
theorem pr_exists_pi_ge {X : Type} [Fintype X] [Nonempty X] (m : ℕ) (A : X → Prop) :
    1 - Real.exp (-(Pr A * m)) ≤ Pr (fun y : Fin m → X => ∃ j, A (y j)) := by
  rw [pr_exists_pi]
  linarith [one_sub_pow_le_exp (Pr_le_one A) m]

/-- An average of an optional value's reading, by the value. -/
theorem ex_elim {X Y : Type} [Fintype X] [Fintype Y] (g : X → Option Y) (F : Y → ℝ) :
    Ex (fun x => (g x).elim 0 F) = ∑ y, Pr (fun x => g x = some y) * F y := by
  classical
  have hpt : ∀ x, (g x).elim 0 F = ∑ y, (if g x = some y then (1 : ℝ) else 0) * F y := by
    intro x
    cases g x with
    | none => simp
    | some y => simp [Finset.sum_ite_eq]
  rw [ex_congr hpt, ex_sum]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [ex_mul_const, ex_ind]

theorem pr_ne_none {X Y : Type} [Fintype X] [Fintype Y] (g : X → Option Y) :
    Pr (fun x => g x ≠ none) = ∑ y, Pr (fun x => g x = some y) := by
  classical
  have h := ex_elim g (fun _ => (1 : ℝ))
  simp only [mul_one] at h
  rw [← h, ← ex_ind]
  exact ex_congr fun x => by cases g x <;> simp

theorem pr_pos_witness {β : Type} [Fintype β] {p : β → Prop} (h : Pr p ≠ 0) : ∃ a, p a := by
  by_contra hne
  exact h ((Pr_congr (q := fun _ => False) fun a => ⟨fun ha => hne ⟨a, ha⟩, False.elim⟩).trans Pr_false)

/-! ## The chase -/

/-- **The chase finds a hit**: where each first play `x` the rule `sel` picks at `(u, b)` is off at `b` (`hoff`), the
off reading at `b` weighs at most the hit (`hmaj`), and the `M b` further plays at `b` carry the rate `β` at `b`'s
weight (`hM`), some further play at the picked bit hits it with probability at least `(1 − e^{−β/2})` times a pick's. -/
theorem chase_success {X ι B : Type} [Fintype X] [Nonempty X] [Fintype ι] [Fintype B] [DecidableEq B]
    (sel : X → Option (ι × B)) (hit off : B → X → Prop) (M : B → ℕ) (β : ℝ)
    (hoff : ∀ x u b, sel x = some (u, b) → off b x) (hmaj : ∀ b, Pr (off b) ≤ Pr (hit b))
    (hM : ∀ b, 0 < Pr (hit b) + Pr (off b) → β ≤ (Pr (hit b) + Pr (off b)) * M b) :
    (1 - Real.exp (-(β / 2))) * Pr (fun x => sel x ≠ none) ≤
      Pr (fun p : X × ((b : B) → Fin (M b) → X) => ∃ ub, sel p.1 = some ub ∧ ∃ j, hit ub.2 (p.2 ub.2 j)) := by
  classical
  have hpt : ∀ x, (sel x).elim 0 (fun ub => Pr (fun y : Fin (M ub.2) → X => ∃ j, hit ub.2 (y j))) =
      Pr (fun T : (b : B) → Fin (M b) → X => ∃ ub, sel x = some ub ∧ ∃ j, hit ub.2 (T ub.2 j)) := by
    intro x
    cases sel x with
    | none => simp [Pr_false]
    | some ub =>
      have e : Pr (fun T : (b : B) → Fin (M b) → X => ∃ ub', some ub = some ub' ∧ ∃ j, hit ub'.2 (T ub'.2 j)) =
          Pr (fun T : (b : B) → Fin (M b) → X => ∃ j, hit ub.2 (T ub.2 j)) :=
        Pr_congr fun T => ⟨fun ⟨ub', h', hj⟩ => by cases h'; exact hj, fun hj => ⟨ub, rfl, hj⟩⟩
      rw [e]
      exact (pr_pi_eval (Smp := fun b => Fin (M b) → X) ub.2 (fun y => ∃ j, hit ub.2 (y j))).symm
  have hA : ∀ ub : ι × B, (1 - Real.exp (-(β / 2))) * Pr (fun x => sel x = some ub) ≤
      Pr (fun x => sel x = some ub) * Pr (fun y : Fin (M ub.2) → X => ∃ j, hit ub.2 (y j)) := by
    rintro ⟨u, b⟩
    by_cases hA0 : Pr (fun x => sel x = some (u, b)) = 0
    · rw [hA0]; simp
    have hApos : 0 < Pr (fun x => sel x = some (u, b)) := lt_of_le_of_ne (Pr_nonneg _) (Ne.symm hA0)
    have hoffpos : 0 < Pr (off b) := lt_of_lt_of_le hApos (Pr_mono fun x hx => hoff x u b hx)
    have hβ := hM b (by linarith [Pr_nonneg (hit b)])
    have hmM := mul_le_mul_of_nonneg_right (hmaj b) (Nat.cast_nonneg (M b) : (0 : ℝ) ≤ M b)
    have hw : β / 2 ≤ Pr (hit b) * M b := by linarith
    have h1 := pr_exists_pi_ge (M b) (hit b)
    have h2 : Real.exp (-(Pr (hit b) * M b)) ≤ Real.exp (-(β / 2)) := Real.exp_le_exp.2 (by linarith)
    have hPge : 1 - Real.exp (-(β / 2)) ≤ Pr (fun y : Fin (M b) → X => ∃ j, hit b (y j)) := by linarith
    rw [mul_comm]
    exact mul_le_mul_of_nonneg_left hPge (Pr_nonneg _)
  calc (1 - Real.exp (-(β / 2))) * Pr (fun x => sel x ≠ none)
      = ∑ ub, (1 - Real.exp (-(β / 2))) * Pr (fun x => sel x = some ub) := by
        rw [pr_ne_none, Finset.mul_sum]
    _ ≤ ∑ ub, Pr (fun x => sel x = some ub) * Pr (fun y : Fin (M ub.2) → X => ∃ j, hit ub.2 (y j)) :=
        Finset.sum_le_sum fun ub _ => hA ub
    _ = Ex (fun x => (sel x).elim 0 (fun ub => Pr (fun y : Fin (M ub.2) → X => ∃ j, hit ub.2 (y j)))) :=
        (ex_elim sel _).symm
    _ = Ex (fun x => Pr (fun T : (b : B) → Fin (M b) → X => ∃ ub, sel x = some ub ∧ ∃ j, hit ub.2 (T ub.2 j))) :=
        ex_congr hpt
    _ = _ := pr_prod (fun x (T : (b : B) → Fin (M b) → X) => ∃ ub, sel x = some ub ∧ ∃ j, hit ub.2 (T ub.2 j))

/-- **The chase's expected cost**: a first play, and where `sel` picks a unit `u` and a bit `b`, `M b` more, is at most
`2 + β n` plays in expectation, `n` the units, where a pick draws its unit (`hdrawn`), a picked unit weighs at most its
bit's `q` (`hq`) and `M b ≤ β / q b + 1` (`hM`). -/
theorem chase_cost {X ι B : Type} [Fintype X] [Nonempty X] [Fintype ι] [Fintype B] [DecidableEq B]
    (sel : X → Option (ι × B)) (drawn : ι → X → Prop) (q : B → ℝ) (M : B → ℕ) (β : ℝ) (pc : X → ℝ)
    (hβ : 0 ≤ β) (hpc : ∀ x, 0 ≤ pc x) (hdrawn : ∀ x u b, sel x = some (u, b) → drawn u x)
    (hq : ∀ x u b, sel x = some (u, b) → Pr (drawn u) ≤ q b) (hM : ∀ b, (M b : ℝ) ≤ β / q b + 1) :
    Ex (fun p : X × ((b : B) → Fin (M b) → X) => pc p.1 + (sel p.1).elim 0 (fun ub => ∑ j, pc (p.2 ub.2 j))) ≤
      (2 + β * Fintype.card ι) * Ex pc := by
  classical
  have hE0 : 0 ≤ Ex pc := Ex_nonneg hpc
  have hrest : ∀ x, Ex (fun T : (b : B) → Fin (M b) → X => (sel x).elim 0 (fun ub => ∑ j, pc (T ub.2 j))) =
      (sel x).elim 0 (fun ub => (M ub.2 : ℝ) * Ex pc) := by
    intro x
    cases sel x with
    | none => simp [Ex_const]
    | some ub =>
      simp only [Option.elim]
      rw [ex_sum]
      have h : ∀ j : Fin (M ub.2), Ex (fun T : (b : B) → Fin (M b) → X => pc (T ub.2 j)) = Ex pc := fun j => by
        rw [ex_pi_eval (Smp := fun b => Fin (M b) → X) ub.2 (fun y => pc (y j))]
        exact ex_pi_eval (Smp := fun _ : Fin (M ub.2) => X) j pc
      rw [Finset.sum_congr rfl fun j _ => h j]
      simp
  have hterm : ∀ ub : ι × B, Pr (fun x => sel x = some ub) * (M ub.2 : ℝ) ≤
      Pr (fun x => sel x = some ub) + β * (Pr (fun x => sel x = some ub) / Pr (drawn ub.1)) := by
    rintro ⟨u, b⟩
    show Pr (fun x => sel x = some (u, b)) * (M b : ℝ) ≤
      Pr (fun x => sel x = some (u, b)) + β * (Pr (fun x => sel x = some (u, b)) / Pr (drawn u))
    by_cases hA0 : Pr (fun x => sel x = some (u, b)) = 0
    · rw [hA0]; simp
    have hApos : 0 < Pr (fun x => sel x = some (u, b)) := lt_of_le_of_ne (Pr_nonneg _) (Ne.symm hA0)
    obtain ⟨x, hx⟩ := pr_pos_witness hA0
    have hpu : 0 < Pr (drawn u) := lt_of_lt_of_le hApos (Pr_mono fun x hx => hdrawn x u b hx)
    have h2 : β / q b ≤ β / Pr (drawn u) := div_le_div_of_nonneg_left hβ hpu (hq x u b hx)
    calc Pr (fun x => sel x = some (u, b)) * (M b : ℝ)
        ≤ Pr (fun x => sel x = some (u, b)) * (β / Pr (drawn u) + 1) :=
          mul_le_mul_of_nonneg_left ((hM b).trans (by linarith)) (Pr_nonneg _)
      _ = _ := by ring
  have hU : ∀ u : ι, ∑ b : B, Pr (fun x => sel x = some (u, b)) / Pr (drawn u) ≤ 1 := by
    intro u
    rw [← Finset.sum_div]
    refine div_le_one_of_le₀ ?_ (Pr_nonneg _)
    rw [Finset.sum_congr rfl fun b _ => (ex_ind (fun x => sel x = some (u, b))).symm, ← ex_sum,
      ← ex_ind (drawn u)]
    refine Ex_mono fun x => ?_
    cases h : sel x with
    | none => simp; positivity
    | some ub =>
      obtain ⟨u', b'⟩ := ub
      by_cases hu : u' = u
      · subst hu
        simp [hdrawn x u' b' h, Finset.sum_ite_eq]
      · simp [hu]; positivity
  have hsel : ∑ ub : ι × B, Pr (fun x => sel x = some ub) * (M ub.2 : ℝ) ≤ 1 + β * Fintype.card ι := by
    calc ∑ ub : ι × B, Pr (fun x => sel x = some ub) * (M ub.2 : ℝ)
        ≤ ∑ ub : ι × B, (Pr (fun x => sel x = some ub) +
            β * (Pr (fun x => sel x = some ub) / Pr (drawn ub.1))) :=
          Finset.sum_le_sum fun ub _ => hterm ub
      _ = Pr (fun x => sel x ≠ none) + β * ∑ u : ι, ∑ b : B, Pr (fun x => sel x = some (u, b)) / Pr (drawn u) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, pr_ne_none]
          congr 2
          exact Fintype.sum_prod_type _
      _ ≤ 1 + β * ∑ _u : ι, (1 : ℝ) :=
          add_le_add (Pr_le_one _) (mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun u _ => hU u) hβ)
      _ = 1 + β * Fintype.card ι := by simp
  calc Ex (fun p : X × ((b : B) → Fin (M b) → X) => pc p.1 + (sel p.1).elim 0 (fun ub => ∑ j, pc (p.2 ub.2 j)))
      = Ex pc + Ex (fun x => (sel x).elim 0 (fun ub => (M ub.2 : ℝ) * Ex pc)) := by
        rw [Ex_add, ex_fst, ex_prod (fun x (T : (b : B) → Fin (M b) → X) =>
          (sel x).elim 0 (fun ub => ∑ j, pc (T ub.2 j))), ex_congr hrest]
    _ = Ex pc + (∑ ub, Pr (fun x => sel x = some ub) * (M ub.2 : ℝ)) * Ex pc := by
        rw [ex_elim, Finset.sum_mul]
        congr 1
        exact Finset.sum_congr rfl fun ub _ => by ring
    _ ≤ Ex pc + (1 + β * Fintype.card ι) * Ex pc := add_le_add le_rfl (mul_le_mul_of_nonneg_right hsel hE0)
    _ = (2 + β * Fintype.card ι) * Ex pc := by ring

end Target

open Target
open Claims (Ex Pr Ex_nonneg Ex_mono Ex_div Pr_nonneg Pr_mono Pr_le_one Pr_false)
open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-! ## The chasing finder's plays -/

/-- **A play's witness** is the extractor's, at an accepted run, in the recursive relation. -/
theorem playWit_some (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    {ξ : S.Ξ} {s : q.Sample V S ξ} {wt : Claims.Flock.Wit} (h : q.playWit V S ξ s = some wt) :
    (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
      (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt := by
  unfold Dep.playWit at h
  by_cases hacc : (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc
  · rw [if_pos hacc] at h
    obtain ⟨he, hR⟩ := Option.filter_eq_some_iff.1 h
    exact ⟨hacc, he, of_decide_eq_true hR⟩
  · rw [if_neg hacc] at h
    cases h

theorem playWit_of (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    {ξ : S.Ξ} {s : q.Sample V S ξ} {wt : Claims.Flock.Wit}
    (hacc : (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc)
    (he : (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt) (hR : q.R (S.x ξ) wt) :
    q.playWit V S ξ s = some wt := by
  unfold Dep.playWit
  rw [if_pos hacc, he]
  exact Option.filter_eq_some_iff.2 ⟨rfl, decide_eq_true hR⟩

/-- **The bit a play is off at**: where `chaseSel` picks the unit `u` and the bit `b`, the play has a witness, draws
`u`, `b`'s wire is `u`'s, and the witness reads the bit the plurality didn't pick there. -/
theorem chaseSel_spec (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) {ξ : S.Ξ} {s : q.Sample V S ξ}
    {u : Fin (V.units θ F)} {b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)}
    (h : q.chaseSel V S readX ξ s = some (u, b)) :
    ∃ wt, q.playWit V S ξ s = some wt ∧ u ∈ S.drawOf ξ ∧ b.1 ∈ (V.partsOf θ F).io u ∧
      (readX (S.x ξ) wt b.1).getLsbD b.2 = !(q.extract V S readX b.1).getLsbD b.2 := by
  unfold Dep.chaseSel at h
  obtain ⟨wt, hw, hf⟩ := Option.bind_eq_some_iff.1 h
  exact ⟨wt, hw, of_decide_eq_true (List.find?_some hf)⟩

theorem chaseSel_off (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) {ξ : S.Ξ} {s : q.Sample V S ξ}
    {u : Fin (V.units θ F)} {b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)}
    (h : q.chaseSel V S readX ξ s = some (u, b)) : bitEvent q V S readX b ξ s := by
  obtain ⟨wt, hw, hu, hg, hb⟩ := chaseSel_spec q V S readX h
  obtain ⟨hacc, he, hR⟩ := playWit_some q V S hw
  exact ⟨hacc, ⟨u, hu, hg⟩, wt, he, hR, hb⟩

theorem chaseSel_drawn (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) {ξ : S.Ξ} {s : q.Sample V S ξ}
    {u : Fin (V.units θ F)} {b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)}
    (h : q.chaseSel V S readX ξ s = some (u, b)) : drawEvent q V S u ξ s := by
  obtain ⟨wt, hw, hu, -, -⟩ := chaseSel_spec q V S readX h
  obtain ⟨hacc, he, hR⟩ := playWit_some q V S hw
  exact ⟨hacc, hu, wt, he, hR⟩

/-- **A play off the plurality is picked**: where a play's witness is off the plurality at a drawn unit, `chaseSel`
picks a unit and a bit. -/
theorem chaseSel_ne_none (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) {ξ : S.Ξ} {s : q.Sample V S ξ}
    (h : ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s) : q.chaseSel V S readX ξ s ≠ none := by
  obtain ⟨u, hacc, hu, wt, he, hR, g, hg, i, hi, hbit⟩ := h
  unfold Dep.chaseSel
  rw [playWit_of q V S hacc he hR, Option.bind_some]
  intro hnone
  refine List.find?_eq_none.1 hnone (u, ⟨g, ⟨i, hi⟩⟩) ?_ (decide_eq_true ⟨hu, hg, hbit⟩)
  exact List.mem_flatMap.2 ⟨u, List.mem_finRange u, List.mem_flatMap.2 ⟨g, List.mem_finRange g,
    List.mem_map.2 ⟨⟨i, hi⟩, List.mem_finRange _, rfl⟩⟩⟩

theorem chaseHit_some (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    {b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)} {t : q.Play V S} {w : Claims.Flock.Wit}
    (h : q.chaseHit V S readX b t = some w) :
    q.playWit V S t.1 (t.2 t.1) = some w ∧ (∃ u ∈ S.drawOf t.1, b.1 ∈ (V.partsOf θ F).io u) ∧
      (readX (S.x t.1) w b.1).getLsbD b.2 = (q.extract V S readX b.1).getLsbD b.2 := by
  unfold Dep.chaseHit at h
  obtain ⟨h1, h2⟩ := Option.filter_eq_some_iff.1 h
  exact ⟨h1, of_decide_eq_true h2⟩

/-- **A further play hits the bit `b`** exactly where it reads the plurality's bit there (`wEvent`). -/
theorem chaseHit_ne_none_iff (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) (t : q.Play V S) :
    q.chaseHit V S readX b t ≠ none ↔
      wEvent q V S readX b.1 b.2 ((q.extract V S readX b.1).getLsbD b.2) t.1 (t.2 t.1) := by
  constructor
  · intro h
    obtain ⟨w, hw⟩ := Option.ne_none_iff_exists'.1 h
    obtain ⟨hpw, hd, hb⟩ := chaseHit_some q V S readX hw
    obtain ⟨hacc, he, hR⟩ := playWit_some q V S hpw
    exact ⟨hacc, hd, w, he, hR, hb⟩
  · rintro ⟨hacc, hd, w, he, hR, hb⟩
    have hw : q.chaseHit V S readX b t = some w := by
      unfold Dep.chaseHit
      rw [playWit_of q V S hacc he hR]
      exact Option.filter_eq_some_iff.2 ⟨rfl, decide_eq_true ⟨hd, hb⟩⟩
    rw [hw]
    simp

open Classical in
/-- **A play hits `b`** with probability `b`'s weight at the plurality's bit. -/
theorem pr_hit_eq (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) :
    Pr (fun t : q.Play V S => q.chaseHit V S readX b t ≠ none) =
      q.weight V S readX b.1 b.2 ((q.extract V S readX b.1).getLsbD b.2) :=
  (Claims.Pr_congr fun t => chaseHit_ne_none_iff q V S readX b t).trans
    (pr_table (Smp := q.Sample V S) fun ξ s =>
      wEvent q V S readX b.1 b.2 ((q.extract V S readX b.1).getLsbD b.2) ξ s)

open Classical in
/-- **A play is off at `b`** with probability `b`'s weight at the bit the plurality didn't pick. -/
theorem pr_off_eq (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) :
    Pr (fun t : q.Play V S => bitEvent q V S readX b t.1 (t.2 t.1)) =
      q.weight V S readX b.1 b.2 (!(q.extract V S readX b.1).getLsbD b.2) :=
  pr_table (Smp := q.Sample V S) fun ξ s => bitEvent q V S readX b ξ s

open Classical in
theorem hit_add_off (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g)) :
    Pr (fun t : q.Play V S => q.chaseHit V S readX b t ≠ none) +
        Pr (fun t : q.Play V S => bitEvent q V S readX b t.1 (t.2 t.1)) =
      q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false := by
  rw [pr_hit_eq, pr_off_eq]
  cases hb : (q.extract V S readX b.1).getLsbD b.2
  · simp only [Bool.not_false]
    exact add_comm _ _
  · simp only [Bool.not_true]

/-- **`chaseM`'s rate**: at a bit some play reads, `chaseM b` plays carry `β` at the bit's weight. -/
theorem chaseM_rate (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript)
    (b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g))
    (hpos : 0 < q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false) :
    chaseBeta (V.units θ F) ≤
      (q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false) * q.chaseM V S readX b := by
  calc chaseBeta (V.units θ F)
      ≤ (q.chaseM V S readX b : ℝ) * (q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false) :=
        (div_le_iff₀ hpos).1 (Nat.le_ceil _)
    _ = _ := mul_comm _ _

/-- **The chasing finder's collision**: where its first play is off at the unit `u` and the bit `b` and one of its
further plays at `b` hits `b`, `collOf` at the first play's witness and the first such further play's outputs a
collision (`hcoll`: the two read `b`'s wire differently, at units the two draw). -/
theorem chase_coll (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', ((collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂).eval FlockDeploy.sha512).1 = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (p : q.Play V S × q.ChasePlays V S readX)
    (h : ∃ ub : Fin (V.units θ F) × Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g),
      q.chaseSel V S readX p.1.1 (p.1.2 p.1.1) = some ub ∧ ∃ j, q.chaseHit V S readX ub.2 (p.2 ub.2 j) ≠ none) :
    ∃ m m', (q.chaseAt V S readX collOf).out p = some (m, m') ∧ Claims.Collision FlockDeploy.sha512 m m' := by
  obtain ⟨⟨u, b⟩, hsel, j, hj⟩ := h
  obtain ⟨w₀, hw₀, hu, hg, hb₀⟩ := chaseSel_spec q V S readX hsel
  obtain ⟨-, -, hR₀⟩ := playWit_some q V S hw₀
  obtain ⟨wj, hwj⟩ := Option.ne_none_iff_exists'.1 hj
  have hne : ((List.finRange (q.chaseM V S readX b)).filterMap fun j =>
      (q.chaseHit V S readX b (p.2 b j)).map fun w => (S.x (p.2 b j).1, w)).head? ≠ none := by
    intro hnone
    rw [List.head?_eq_none_iff] at hnone
    have hmem : (S.x (p.2 b j).1, wj) ∈ (List.finRange (q.chaseM V S readX b)).filterMap fun j =>
        (q.chaseHit V S readX b (p.2 b j)).map fun w => (S.x (p.2 b j).1, w) :=
      List.mem_filterMap.2 ⟨j, List.mem_finRange j, by simp [hwj]⟩
    rw [hnone] at hmem
    simp at hmem
  obtain ⟨xw, hxw⟩ := Option.ne_none_iff_exists'.1 hne
  obtain ⟨ys, hys⟩ := List.head?_eq_some_iff.1 hxw
  have hmem : xw ∈ (List.finRange (q.chaseM V S readX b)).filterMap fun j =>
      (q.chaseHit V S readX b (p.2 b j)).map fun w => (S.x (p.2 b j).1, w) := by
    rw [hys]
    simp
  obtain ⟨j', -, hj'⟩ := List.mem_filterMap.1 hmem
  obtain ⟨w, hw, rfl⟩ := Option.map_eq_some_iff.1 hj'
  obtain ⟨hpw, hd, hb⟩ := chaseHit_some q V S readX hw
  obtain ⟨-, -, hR⟩ := playWit_some q V S hpw
  have hne' : readX (S.x p.1.1) w₀ b.1 ≠ readX (S.x (p.2 b j').1) w b.1 := fun h => by
    rw [h, hb] at hb₀
    cases hx : (q.extract V S readX b.1).getLsbD b.2 <;> simp [hx] at hb₀
  obtain ⟨m, m', hm, hcol⟩ := hcoll p.1.1 (p.2 b j').1 w₀ w b.1 hR₀ hR ⟨u, hu, hg⟩ hd hne'
  have hpair : q.chasePair V S readX p = some (S.x p.1.1, w₀, S.x (p.2 b j').1, w) := by
    unfold Dep.chasePair
    exact Option.bind_eq_some_iff.2 ⟨(u, b), hsel, Option.bind_eq_some_iff.2 ⟨w₀, hw₀,
      Option.map_eq_some_iff.2 ⟨_, hxw, rfl⟩⟩⟩
  refine ⟨m, m', ?_, hcol⟩
  show (collRun collOf (q.chasePair V S readX p)).1 = some (m, m')
  rw [hpair]
  exact hm

open Classical in
/-- **The chasing finder's expected calls are at most its budget** (`Dep.chaseBudget`): each play's are at most its
budget (`Dep.ex_playCost_le`), the further plays number at most `2 + β n` plays in all in expectation
(`chase_cost`), and `collOf`'s calls are at most `dColl` (`hdColl`). -/
theorem ex_chaseAt_cost_le (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (nI : Msg → ℕ) (dColl : ℕ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hdColl : ∀ x₁ w₁ x₂ w₂, ((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).2.length ≤ dColl) :
    Ex (fun s => ((q.chaseAt V S readX collOf).cost s : ℝ)) ≤ q.chaseBudget V S nI dColl := by
  have hβ0 : 0 ≤ chaseBeta (V.units θ F) := Real.sqrt_nonneg _
  have hw0 : ∀ b : Σ g : Fin (V.circuitOf θ F).N, Fin ((V.circuitOf θ F).w g),
      0 ≤ q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false := fun b =>
    add_nonneg (Ex_nonneg fun _ => Pr_nonneg _) (Ex_nonneg fun _ => Pr_nonneg _)
  have hc := chase_cost (X := q.Play V S) (fun x => q.chaseSel V S readX x.1 (x.2 x.1))
    (fun u x => drawEvent q V S u x.1 (x.2 x.1))
    (fun b => q.weight V S readX b.1 b.2 true + q.weight V S readX b.1 b.2 false) (q.chaseM V S readX)
    (chaseBeta (V.units θ F)) (fun x => (q.playCost V S x.1 (x.2 x.1) : ℝ)) hβ0 (fun _ => Nat.cast_nonneg _)
    (fun x u b h => chaseSel_drawn q V S readX h)
    (fun x u b h => (pr_table (Smp := q.Sample V S) fun ξ s => drawEvent q V S u ξ s).trans_le (by
      obtain ⟨wt, -, -, hg, -⟩ := chaseSel_spec q V S readX h
      exact drawn_le_sum q V S readX hg b.2))
    (fun b => (Nat.ceil_lt_add_one (div_nonneg hβ0 (hw0 b))).le)
  rw [Fintype.card_fin] at hc
  have hpt : ∀ p : q.Play V S × q.ChasePlays V S readX, ((q.chaseAt V S readX collOf).cost p : ℝ) ≤
      ((q.playCost V S p.1.1 (p.1.2 p.1.1) : ℝ) + (q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0
        (fun ub => ∑ j : Fin (q.chaseM V S readX ub.2),
          (q.playCost V S (p.2 ub.2 j).1 ((p.2 ub.2 j).2 (p.2 ub.2 j).1) : ℝ))) + dColl := by
    intro p
    have h := collRun_calls_le collOf dColl hdColl (q.chasePair V S readX p)
    have he : ((((q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0 fun b =>
        ∑ j : Fin (q.chaseM V S readX b.2), q.playCost V S (p.2 b.2 j).1 ((p.2 b.2 j).2 (p.2 b.2 j).1)) : ℕ) : ℝ) =
        (q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0 (fun ub => ∑ j : Fin (q.chaseM V S readX ub.2),
          (q.playCost V S (p.2 ub.2 j).1 ((p.2 ub.2 j).2 (p.2 ub.2 j).1) : ℝ)) := by
      cases q.chaseSel V S readX p.1.1 (p.1.2 p.1.1) <;> simp
    show ((q.playCost V S p.1.1 (p.1.2 p.1.1) + ((q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0 fun b =>
        ∑ j : Fin (q.chaseM V S readX b.2), q.playCost V S (p.2 b.2 j).1 ((p.2 b.2 j).2 (p.2 b.2 j).1)) +
        (collRun collOf (q.chasePair V S readX p)).2 : ℕ) : ℝ) ≤ _
    rw [Nat.cast_add, Nat.cast_add, he]
    exact add_le_add le_rfl (by exact_mod_cast h)
  have hE : Ex (fun x : q.Play V S => (q.playCost V S x.1 (x.2 x.1) : ℝ)) ≤
      Ex (fun ξ : S.Ξ => (q.playBudget V S nI ξ : ℝ)) :=
    (ex_table (Smp := q.Sample V S) fun ξ s => (q.playCost V S ξ s : ℝ)).trans_le
      (Ex_mono fun ξ => q.ex_playCost_le V S nI hn ξ)
  have hb2 : 0 ≤ 2 + chaseBeta (V.units θ F) * V.units θ F :=
    add_nonneg (by norm_num) (mul_nonneg hβ0 (Nat.cast_nonneg _))
  calc Ex (fun s => ((q.chaseAt V S readX collOf).cost s : ℝ))
      ≤ Ex (fun p : q.Play V S × q.ChasePlays V S readX =>
          ((q.playCost V S p.1.1 (p.1.2 p.1.1) : ℝ) + (q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0
            (fun ub => ∑ j : Fin (q.chaseM V S readX ub.2),
              (q.playCost V S (p.2 ub.2 j).1 ((p.2 ub.2 j).2 (p.2 ub.2 j).1) : ℝ))) + (dColl : ℝ)) :=
        Ex_mono hpt
    _ = Ex (fun p : q.Play V S × q.ChasePlays V S readX =>
          (q.playCost V S p.1.1 (p.1.2 p.1.1) : ℝ) + (q.chaseSel V S readX p.1.1 (p.1.2 p.1.1)).elim 0
            (fun ub => ∑ j : Fin (q.chaseM V S readX ub.2),
              (q.playCost V S (p.2 ub.2 j).1 ((p.2 ub.2 j).2 (p.2 ub.2 j).1) : ℝ))) + dColl :=
        ex_add_const _ _
    _ ≤ (2 + chaseBeta (V.units θ F) * V.units θ F) *
          Ex (fun x : q.Play V S => (q.playCost V S x.1 (x.2 x.1) : ℝ)) + dColl := add_le_add hc le_rfl
    _ ≤ (2 + chaseBeta (V.units θ F) * V.units θ F) * Ex (fun ξ : S.Ξ => (q.playBudget V S nI ξ : ℝ)) + dColl :=
        add_le_add (mul_le_mul_of_nonneg_left hE hb2) le_rfl
    _ = q.chaseBudget V S nI dColl := rfl

open Classical in
/-- **The binding term at the chasing finder**: averaged over the draws, the probability that the run accepts while
the extractor's witness differs from the plurality `Dep.extract` at a bit of a wire of a drawn unit is at most
`chaseBudget / ((1 − e^{−β/2}) · 2^256)` under `ChaseCR`'s binding premise (`hchase`): such a play is picked
(`chaseSel_ne_none`), the chasing finder then collides with probability at least `1 − e^{−β/2}` (`chase_success`,
`chase_coll`), and its expected calls are at most `chaseBudget` (`ex_chaseAt_cost_le`). At no units, no play is off. -/
theorem ex_off_le_chase (q : Dep) (V : Vocab d α) {θ c : ByteArray} {F : Array ByteArray}
    {σ : Game.Strategy (V.game θ F c)} {k : Verity.Registration.Composite} (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (nI : Msg → ℕ) (dColl : ℕ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hdColl : ∀ x₁ w₁ x₂ w₂, ((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).2.length ≤ dColl)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', ((collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂).eval FlockDeploy.sha512).1 = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hchase : Claims.ExpectedCR FlockDeploy.sha512 (q.chaseAt V S readX collOf)) :
    Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      q.chaseBudget V S nI dColl / ((1 - Real.exp (-(chaseBeta (V.units θ F) / 2))) * 2 ^ (256 : ℝ)) := by
  have hβ0 : 0 ≤ chaseBeta (V.units θ F) := Real.sqrt_nonneg _
  have h256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have hden0 : 0 ≤ 1 - Real.exp (-(chaseBeta (V.units θ F) / 2)) := by
    have h := Real.exp_le_exp.2 (show -(chaseBeta (V.units θ F) / 2) ≤ 0 by linarith)
    rw [Real.exp_zero] at h
    linarith
  have hb2 : 0 ≤ 2 + chaseBeta (V.units θ F) * V.units θ F :=
    add_nonneg (by norm_num) (mul_nonneg hβ0 (Nat.cast_nonneg _))
  have hB0 : 0 ≤ q.chaseBudget V S nI dColl :=
    add_nonneg (mul_nonneg hb2 (Ex_nonneg fun _ => Nat.cast_nonneg _)) (Nat.cast_nonneg _)
  rcases Nat.eq_zero_or_pos (V.units θ F) with h0 | hpos
  · have hL : ∀ ξ, Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s) = 0 :=
      fun ξ => (Claims.Pr_congr fun s => ⟨fun ⟨u, _⟩ => absurd u.isLt (by omega), False.elim⟩).trans Pr_false
    rw [ex_congr hL, Claims.Ex_const]
    exact div_nonneg hB0 (mul_nonneg hden0 h256.le)
  have hβ : 0 < chaseBeta (V.units θ F) := Real.sqrt_pos.2 (div_pos (by norm_num) (by exact_mod_cast hpos))
  have hden : 0 < 1 - Real.exp (-(chaseBeta (V.units θ F) / 2)) := by
    have h := Real.exp_lt_exp.2 (show -(chaseBeta (V.units θ F) / 2) < 0 by linarith)
    rw [Real.exp_zero] at h
    linarith
  have hoff : ∀ (x : q.Play V S) u b, q.chaseSel V S readX x.1 (x.2 x.1) = some (u, b) →
      bitEvent q V S readX b x.1 (x.2 x.1) := fun x u b h => chaseSel_off q V S readX h
  have hmaj : ∀ b, Pr (fun x : q.Play V S => bitEvent q V S readX b x.1 (x.2 x.1)) ≤
      Pr (fun x : q.Play V S => q.chaseHit V S readX b x ≠ none) := fun b =>
    (pr_off_eq q V S readX b).trans_le ((minority_le_majority q V S readX b.1 b.2.isLt).trans_eq
      (pr_hit_eq q V S readX b).symm)
  have hM : ∀ b, 0 < Pr (fun x : q.Play V S => q.chaseHit V S readX b x ≠ none) +
        Pr (fun x : q.Play V S => bitEvent q V S readX b x.1 (x.2 x.1)) →
      chaseBeta (V.units θ F) ≤ (Pr (fun x : q.Play V S => q.chaseHit V S readX b x ≠ none) +
        Pr (fun x : q.Play V S => bitEvent q V S readX b x.1 (x.2 x.1))) * q.chaseM V S readX b := by
    intro b hp
    rw [hit_add_off q V S readX b] at hp ⊢
    exact chaseM_rate q V S readX b hp
  have hsucc := chase_success (fun x : q.Play V S => q.chaseSel V S readX x.1 (x.2 x.1))
    (fun b x => q.chaseHit V S readX b x ≠ none) (fun b x => bitEvent q V S readX b x.1 (x.2 x.1))
    (q.chaseM V S readX) (chaseBeta (V.units θ F)) hoff hmaj hM
  have hfind : Pr (fun p : q.Play V S × q.ChasePlays V S readX =>
      ∃ ub, q.chaseSel V S readX p.1.1 (p.1.2 p.1.1) = some ub ∧
        ∃ j, q.chaseHit V S readX ub.2 (p.2 ub.2 j) ≠ none) ≤
      (q.chaseAt V S readX collOf).Pr FlockDeploy.sha512 :=
    Pr_mono fun p hp => chase_coll q V S readX collOf hcoll p hp
  have hP0 : Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Ex (fun ξ => Pr (fun s : q.Sample V S ξ => q.chaseSel V S readX ξ s ≠ none)) :=
    Ex_mono fun ξ => Pr_mono fun s hs => chaseSel_ne_none q V S readX hs
  have hP : Ex (fun ξ => Pr (fun s : q.Sample V S ξ => ∃ u, offEvent q V S readX (q.extract V S readX) u ξ s)) ≤
      Pr (fun x : q.Play V S => q.chaseSel V S readX x.1 (x.2 x.1) ≠ none) :=
    hP0.trans_eq (pr_table (Smp := q.Sample V S) fun ξ s => q.chaseSel V S readX ξ s ≠ none).symm
  have h1 : (1 - Real.exp (-(chaseBeta (V.units θ F) / 2))) *
      Pr (fun x : q.Play V S => q.chaseSel V S readX x.1 (x.2 x.1) ≠ none) ≤
      q.chaseBudget V S nI dColl / 2 ^ (256 : ℝ) :=
    le_trans hsucc (le_trans hfind (le_trans hchase (div_le_div_of_nonneg_right
      (ex_chaseAt_cost_le q V S readX collOf nI dColl hn hdColl) h256.le)))
  have h2 := (le_div_iff₀ h256).1 h1
  rw [le_div_iff₀ (mul_pos hden h256)]
  calc _ ≤ Pr (fun x : q.Play V S => q.chaseSel V S readX x.1 (x.2 x.1) ≠ none) *
          ((1 - Real.exp (-(chaseBeta (V.units θ F) / 2))) * 2 ^ (256 : ℝ)) :=
        mul_le_mul_of_nonneg_right hP (mul_pos hden h256).le
    _ = (1 - Real.exp (-(chaseBeta (V.units θ F) / 2))) *
          Pr (fun x : q.Play V S => q.chaseSel V S readX x.1 (x.2 x.1) ≠ none) * 2 ^ (256 : ℝ) := by ring
    _ ≤ q.chaseBudget V S nI dColl := h2

/-! ## The target -/

open Classical in
/-- **The target's bound at the chasing finder**: the probability that the recursive component accepts while sampled
proofs' claim fails on the extractor's transcript is at most `Dep.deltaChase`,
`δs + t' · 2^{-λ} + chaseBudget / ((1 − e^{−β/2}) · 2^256)`, where the finders built from the strategy meet `ChaseCR`,
and `⊤` elsewhere (`prob_le_off`, with `ex_off_le_chase`). It takes `sampledProofsRecursive`'s hypotheses. -/
theorem sampledProofsRecursive_chase (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (dColl K₀ : ℕ) (δs : ℝ)
    (hK₀ : ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m)
    (hdraw : ∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs)
    (hX : V.extract θ F c σ = q.extract V S readX)
    (hread : ∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hdColl : ∀ x₁ w₁ x₂ w₂, ((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).2.length ≤ dColl)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', ((collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂).eval FlockDeploy.sha512).1 = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      if q.ChaseCR V S readX collOf nI vS then q.deltaChase V S nI dColl δs lam else ⊤ := by
  split_ifs with hcr
  swap
  · exact le_top
  refine le_trans (prob_le_off q hKS hdecI hdecO nI vS lam hn hv hlam V k θ F c σ S readX hcr.1 K₀ δs hK₀ hdraw hX
    hread hlive) ?_
  unfold Dep.deltaChase
  exact ENNReal.ofReal_le_ofReal (add_le_add le_rfl
    (ex_off_le_chase q V S readX collOf nI dColl hn hdColl hcoll hcr.2))

/-- **The target: sampled proofs at the recursive configuration**, in the per-prover form over the causal game (R4′,
R5, R6), with the recursive configuration's verdict (`RecVerdict`). At composite `k`, the parameters `θ`, the
committed files `F`, the commitment `c` and a strategy `σ`, with `S` its recursive session (`SessionIn`), the
probability that the recursive component accepts while sampled proofs' claim fails on the extractor's transcript
(`¬ Holds k .sampled`) is at most `Dep.errAt` at `S`: the smaller of `Dep.delta` where the finders built from `σ` meet
`FinderCR` (`⊤` elsewhere), `δs + t'(σ) · 2^{-λ} + √(min(bits, 2n) · E[bindBudget(σ)] / 2^256)`, and
`Dep.deltaChase` where they meet `ChaseCR` (`⊤` elsewhere), `δs + t'(σ) · 2^{-λ} + chaseBudget(σ) /
((1 − e^{−β/2}) · 2^256)` at `β = √(8/n)`, with `n` the units, `t'(σ)` the strategy's SHA-512 calls in one run
(`Vocab.cost`) and every finder's budget defined from it and public counts. There is no hypothesis on the strategy but
`live-verifier`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings (`hKS`);
* the decoders meet the list-decoding spec (`hdecI`, `hdecO`);
* the inner and setup call bounds `nI`, `vS` and the level `λ` (`hn`, `hv`, `hlam`), at the statements some run of the
  deployed recursive verifier accepts (`Dep.RunAccepts`);
* `K₀` covers the claim's `ε` share (`hK₀`), and the session's draw misses any `K₀` units with probability at most `δs`
  (`hdraw`);
* the vocabulary's extractor is the recursive configuration's (`hX`, `Dep.extract` at `S`);
* the deployment's reading `readX` of a witness in the recursive relation is correct at, and opened by the commitment
  on, every unit the statement draws (`hread`); `collOf`, an oracle program at SHA-512, makes at most `dColl` calls
  (`hdColl`) and, from two such witnesses that read a wire of units drawn in both differently, outputs a SHA-512
  collision (`hcoll`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until the
  component reads the coins through the coin server's signed device log; then this hypothesis goes.

Each term is its finder's bound (`sampledProofsRecursive_bind`, `sampledProofsRecursive_chase`). -/
theorem sampledProofsRecursive (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, q.RunAccepts x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, q.RunAccepts x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, q.RunAccepts x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α) (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (S : SessionIn V q k θ F c σ)
    (readX : Msg → Claims.Flock.Wit → (V.circuitOf θ F).Transcript) (collOf : CollOf) (dColl K₀ : ℕ) (δs : ℝ)
    (hK₀ : ∀ m : ℕ, (V.certOf θ).ε * V.units θ F < m → K₀ ≤ m)
    (hdraw : ∀ U : Finset (Fin (V.units θ F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs)
    (hX : V.extract θ F c σ = q.extract V S readX)
    (hread : ∀ ξ wt, V.Registered θ F → q.R (S.x ξ) wt → ∀ u ∈ S.drawOf ξ,
      (V.partsOf θ F).Correct (readX (S.x ξ) wt) u ∧ ∀ g ∈ (V.partsOf θ F).io u, V.Opens θ F c g (readX (S.x ξ) wt g))
    (hdColl : ∀ x₁ w₁ x₂ w₂, ((collOf x₁ w₁ x₂ w₂).eval FlockDeploy.sha512).2.length ≤ dColl)
    (hcoll : ∀ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf θ F).N), q.R (S.x ξ₁) w₁ → q.R (S.x ξ₂) w₂ →
      (∃ u ∈ S.drawOf ξ₁, g ∈ (V.partsOf θ F).io u) → (∃ u ∈ S.drawOf ξ₂, g ∈ (V.partsOf θ F).io u) →
      readX (S.x ξ₁) w₁ g ≠ readX (S.x ξ₂) w₂ g →
      ∃ m m', ((collOf (S.x ξ₁) w₁ (S.x ξ₂) w₂).eval FlockDeploy.sha512).1 = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ ξ ω r, LiveCoins q.Bd.IVStar q.st (S.x ξ)
      (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) ω r).seen) :
    Game.prob (fun o => RecVerdict V k o ∧ ¬ V.Holds k θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.errAt V S readX collOf dColl nI vS δs lam := by
  unfold Dep.errAt
  exact le_min
    (sampledProofsRecursive_bind q hKS hdecI hdecO nI vS lam hn hv hlam V k θ F c σ S readX collOf dColl K₀ δs hK₀
      hdraw hX hread hdColl hcoll hlive)
    (sampledProofsRecursive_chase q hKS hdecI hdecO nI vS lam hn hv hlam V k θ F c σ S readX collOf dColl K₀ δs hK₀
      hdraw hX hread hdColl hcoll hlive)

end Verity.Recursive
