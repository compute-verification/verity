module

public import Proofs.Flock.Claims.Prob
public import Proofs.Core.Circuit
public import Definitions.Accounting.JointLive
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# The majority: the transcript a sampled leg extracts, and the binding term

Both sampled legs that read a transcript off many plays (the recursive leg, `Verity.Recursive.spotRecursive`, and the
two-stage leg, `Verity.SampledTwo`) extract it the same way and bound the same event. This module holds that argument
once, over a structure of only what it reads (`Plays`), and each leg instantiates it.

* **The plays** (`Plays C n P`): over a circuit `C` partitioned into `n` units by `P`, the outer samples `Ξ` (averaged),
  each with a finite space of plays `Smp ξ`; at each play, whether it accepts (`acc`), the units it binds
  (`drawn`), the witness the extractor returns (`wit`), the relation the witness must be in (`R`), and the witness read
  as a transcript of `C` (`read`).
* **The plurality** (`Plays.weight`, `Plays.extract`): each bit of each wire is the value the extracted witnesses in
  the relation read there most often, over the accepted plays that bind a unit holding the wire (`false` on a tie).
* **The binding term**: where a play off the plurality at a bit of a bound unit's wire collides, through `coll`, with
  every play that reads the plurality's bit there (`Plays.Binds`), and two independent plays collide with probability
  at most `K`, the plays off the plurality weigh at most `√(min(W, 2n) · K)`, `W` the circuit's wire bits
  (`Plays.ex_off_le`). Both bounds pick one piece per play (`units_bound`), giving each piece a share `a` at most `p`:
  - at the bits (`Plays.ex_off_le_bits`), `p` is the bit's minority weight, at most its majority's
    (`Plays.minority_le_majority`), so `∑ a p ≤ K` and `(∑ a)² ≤ W ∑ a² ≤ WK`;
  - at the units (`Plays.ex_off_le_units`), `p_u` is the probability of binding `u`, and the majority at any bit of
    `u`'s wires weighs at least `p_u / 2` (`Plays.drawn_le_weight`), so `∑ a p ≤ 2K` and `(∑ a)² ≤ n ∑ a² ≤ 2nK`.
* **A unit bound in a witness and unbound in a transcript** (`unbound_drawn`): the witness differs from the
  transcript at some bit of one of the unit's wires.

The uniform averages the argument reads (`ex_*`, `pr_*`) and the plurality word's bits (`getLsbD_ofNat_sum`) are here
too.
-/

@[expose] public section

namespace Verity.Majority

open Claims

/-! ## Uniform averages -/

theorem ex_ind {β : Type} [Fintype β] (p : β → Prop) [DecidablePred p] :
    Ex (fun a => if p a then (1 : ℝ) else 0) = Pr p := by
  rw [Pr_eq_sum]; rfl

theorem pr_or_le {β : Type} [Fintype β] (p q : β → Prop) : Pr (fun a => p a ∨ q a) ≤ Pr p + Pr q := by
  classical
  rw [← ex_ind, ← ex_ind, ← ex_ind, ← Ex_add]
  refine Ex_mono fun a => ?_
  by_cases hp : p a <;> by_cases hq : q a <;> simp [hp, hq]

/-- The union bound over a finite set. -/
theorem pr_exists_finset_le {β I : Type} [Fintype β] (s : Finset I) (E : I → β → Prop) :
    Pr (fun a => ∃ i ∈ s, E i a) ≤ ∑ i ∈ s, Pr (E i) := by
  classical
  induction s using Finset.induction_on with
  | empty => simp [Pr_false]
  | insert j s hj ih =>
      rw [Finset.sum_insert hj]
      refine le_trans (Pr_mono fun a ha => ?_) ((pr_or_le _ _).trans (add_le_add le_rfl ih))
      obtain ⟨i, hi, he⟩ := ha
      rcases Finset.mem_insert.mp hi with rfl | hi
      · exact Or.inl he
      · exact Or.inr ⟨i, hi, he⟩

theorem ex_add3 {β : Type} [Fintype β] (f h g : β → ℝ) :
    Ex (fun x => f x + h x + g x) = Ex f + Ex h + Ex g := by
  rw [Ex_add (fun x => f x + h x) g, Ex_add f h]

theorem ex_sum {β I : Type} [Fintype β] (s : Finset I) (f : I → β → ℝ) :
    Ex (fun x => ∑ i ∈ s, f i x) = ∑ i ∈ s, Ex (f i) := by
  unfold Ex
  rw [Finset.sum_comm, Finset.sum_div]

theorem ex_congr {β : Type} [Fintype β] {f g : β → ℝ} (h : ∀ a, f a = g a) : Ex f = Ex g := by
  rw [funext h]

theorem ex_mul_const {β : Type} [Fintype β] (f : β → ℝ) (c : ℝ) : Ex (fun a => f a * c) = Ex f * c := by
  unfold Ex; rw [← Finset.sum_mul]; ring

theorem ex_const_mul {β : Type} [Fintype β] (c : ℝ) (f : β → ℝ) : Ex (fun a => c * f a) = c * Ex f := by
  unfold Ex; rw [← Finset.mul_sum]; ring

/-- An average over a product is the iterated average. -/
theorem ex_prod {β γ : Type} [Fintype β] [Fintype γ] (f : β → γ → ℝ) :
    Ex (fun p : β × γ => f p.1 p.2) = Ex (fun a => Ex (fun b => f a b)) := by
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.card_prod, ← Finset.sum_div]
  push_cast
  ring

theorem ex_comm {β γ : Type} [Fintype β] [Fintype γ] (f : β → γ → ℝ) :
    Ex (fun a => Ex (fun b => f a b)) = Ex (fun b => Ex (fun a => f a b)) := by
  rw [← ex_prod, ← ex_prod (fun b a => f a b)]
  unfold Ex
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type, Finset.sum_comm, Fintype.card_prod, Fintype.card_prod, mul_comm]

/-- An event of a product's two coordinates, averaged over the first. -/
theorem pr_prod {β γ : Type} [Fintype β] [Fintype γ] (P : β → γ → Prop) :
    Ex (fun b => Pr (fun c => P b c)) = Pr (fun s : β × γ => P s.1 s.2) := by
  classical
  rw [← ex_ind (fun s : β × γ => P s.1 s.2), ex_prod (fun b c => if P b c then (1 : ℝ) else 0)]
  exact ex_congr fun b => (ex_ind (P b)).symm

/-- The average over one point is the value there. -/
theorem ex_unit (f : Unit → ℝ) : Ex f = f () := by
  unfold Ex; simp

/-- A bound at every pair of samples averages to the averaged bound. -/
theorem ex_pairs_le {Ξ : Type} [Fintype Ξ] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    (coll : (ξ₁ : Ξ) → Smp ξ₁ → (ξ₂ : Ξ) → Smp ξ₂ → Prop) (b : Ξ → Ξ → ℝ) (c : ℝ)
    (h : ∀ ξ₁ ξ₂, Pr (fun s : Smp ξ₁ × Smp ξ₂ => coll ξ₁ s.1 ξ₂ s.2) ≤ b ξ₁ ξ₂ / c) :
    Ex (fun p : Ξ × Ξ => Pr (fun s : Smp p.1 × Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤
      Ex (fun p : Ξ × Ξ => b p.1 p.2) / c := by
  rw [← Ex_div]
  exact Ex_mono fun p => h p.1 p.2

/-! ## The plurality word -/

theorem testBit_sum_pow (p : ℕ → Prop) [DecidablePred p] :
    ∀ n, (∑ j ∈ Finset.range n, if p j then 2 ^ j else 0) < 2 ^ n ∧
      ∀ i < n, ((∑ j ∈ Finset.range n, if p j then 2 ^ j else 0).testBit i = true ↔ p i)
  | 0 => ⟨by simp, fun i hi => absurd hi (Nat.not_lt_zero _)⟩
  | n + 1 => by
    obtain ⟨hlt, hbit⟩ := testBit_sum_pow p n
    rw [Finset.sum_range_succ]
    by_cases hn : p n
    · rw [if_pos hn, Nat.add_comm]
      refine ⟨by rw [Nat.pow_succ]; omega, fun i hi => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
      · rw [Nat.testBit_two_pow_add_gt hi]; exact hbit i hi
      · rw [Nat.testBit_two_pow_add_eq, Nat.testBit_lt_two_pow hlt]; simp [hn]
    · rw [if_neg hn, Nat.add_zero]
      refine ⟨lt_of_lt_of_le hlt (Nat.pow_le_pow_right (by norm_num) (Nat.le_succ n)), fun i hi => ?_⟩
      rcases Nat.lt_succ_iff_lt_or_eq.1 hi with hi | rfl
      · exact hbit i hi
      · rw [Nat.testBit_lt_two_pow hlt]; simp [hn]

/-- Bit `i < w` of the plurality word is set exactly where `p i`. -/
theorem getLsbD_ofNat_sum (w : ℕ) (p : ℕ → Prop) [DecidablePred p] {i : ℕ} (hi : i < w) :
    (BitVec.ofNat w (∑ j ∈ Finset.range w, if p j then 2 ^ j else 0)).getLsbD i = true ↔ p i := by
  rw [BitVec.getLsbD_ofNat, decide_eq_true hi, Bool.true_and]
  exact (testBit_sum_pow p w).2 i hi

/-! ## Counting pieces -/

/-- **The pieces' share**: over plays `(ξ, s)`, `s` uniform on `Smp ξ` given `ξ`, let `off u` be events of the pieces
`u` of a finite `ι`, each within `drawn u`. Where a play in `off u` makes `c` times the probability that a second,
independent play collides with it (`coll`) at least that of `drawn u`, and two independent plays collide with
probability at most `K`, the plays in some `off u` have probability at most `√(c |ι| K)`. One piece picked per play
gives each piece a share `a u ≤ p u = Pr[drawn u]`; the pairs give `∑ a u · p u ≤ cK`, so
`(∑ a u)² ≤ |ι| ∑ a u² ≤ c |ι| K`. -/
theorem units_bound {Ξ ι : Type} [Fintype Ξ] [Fintype ι] {Smp : Ξ → Type} [∀ ξ, Fintype (Smp ξ)]
    (off drawn : ι → (ξ : Ξ) → Smp ξ → Prop) (coll : (ξ₁ : Ξ) → Smp ξ₁ → (ξ₂ : Ξ) → Smp ξ₂ → Prop) {c K : ℝ}
    (hc : 0 ≤ c) (hdrawn : ∀ u ξ s, off u ξ s → drawn u ξ s)
    (hlow : ∀ u ξ₁ (s₁ : Smp ξ₁), off u ξ₁ s₁ →
      Ex (fun ξ => Pr (drawn u ξ)) ≤ c * Ex (fun ξ₂ => Pr (coll ξ₁ s₁ ξ₂)))
    (hK : Ex (fun p : Ξ × Ξ => Pr (fun s : Smp p.1 × Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤ K) :
    Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ Real.sqrt (c * Fintype.card ι * K) := by
  classical
  obtain ⟨sel, hsel, hsel'⟩ : ∃ sel : (ξ : Ξ) → Smp ξ → Option ι,
      (∀ ξ s u, sel ξ s = some u → off u ξ s) ∧ ∀ ξ s, (∃ u, off u ξ s) → ∃ u, sel ξ s = some u := by
    refine ⟨fun ξ s => if h : ∃ u, off u ξ s then some h.choose else none, fun ξ s u h => ?_,
      fun ξ s hex => ⟨hex.choose, dite_eq_left hex⟩⟩
    change (if h : ∃ u, off u ξ s then some h.choose else none) = some u at h
    split_ifs at h with hex
    rw [← Option.some.inj h]
    exact hex.choose_spec
  obtain ⟨a, ha⟩ : ∃ a : ι → ℝ, a = fun u => Ex fun ξ => Pr fun s : Smp ξ => sel ξ s = some u := ⟨_, rfl⟩
  obtain ⟨p, hp⟩ : ∃ p : ι → ℝ, p = fun u => Ex fun ξ => Pr (drawn u ξ) := ⟨_, rfl⟩
  have ha0 : ∀ u, 0 ≤ a u := fun u => by rw [ha]; exact Ex_nonneg fun _ => Pr_nonneg _
  have hap : ∀ u, a u ≤ p u := fun u => by
    rw [ha, hp]; exact Ex_mono fun ξ => Pr_mono fun s hs => hdrawn u ξ s (hsel ξ s u hs)
  have h1 : Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ ∑ u, a u := by
    rw [ha, ← ex_sum Finset.univ (fun u ξ => Pr fun s : Smp ξ => sel ξ s = some u)]
    refine Ex_mono fun ξ => le_trans (Pr_mono fun s hs => ?_) (pr_exists_finset_le Finset.univ _)
    obtain ⟨u, hu⟩ := hsel' ξ s hs
    exact ⟨u, Finset.mem_univ _, hu⟩
  have hpt : ∀ ξ (s : Smp ξ), (∑ u, (if sel ξ s = some u then (1 : ℝ) else 0) * p u) ≤
      c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)) := by
    intro ξ s
    cases h : sel ξ s with
    | none =>
        simp only [reduceCtorEq, ite_false, zero_mul, Finset.sum_const_zero]
        exact mul_nonneg hc (Ex_nonneg fun _ => Pr_nonneg _)
    | some u₀ =>
        rw [Finset.sum_eq_single u₀ (fun u _ hu => by simp [Ne.symm hu]) (fun h' => absurd (Finset.mem_univ u₀) h'),
          show (if some u₀ = some u₀ then (1 : ℝ) else 0) = 1 by simp, one_mul, hp]
        exact hlow u₀ ξ s (hsel ξ s u₀ h)
  have h3 : ∑ u, a u * p u ≤ c * K := by
    have e1 : ∑ u, a u * p u =
        Ex (fun ξ => Ex (fun s : Smp ξ => ∑ u, (if sel ξ s = some u then (1 : ℝ) else 0) * p u)) := by
      rw [ex_congr fun ξ => ex_sum Finset.univ (fun u (s : Smp ξ) => (if sel ξ s = some u then (1 : ℝ) else 0) * p u),
        ex_sum]
      refine Finset.sum_congr rfl fun u _ => ?_
      rw [ha, ← ex_mul_const]
      exact ex_congr fun ξ => by rw [ex_mul_const, ex_ind]
    have e2 : Ex (fun ξ => Ex (fun s : Smp ξ => c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)))) =
        c * Ex (fun p : Ξ × Ξ => Pr (fun s : Smp p.1 × Smp p.2 => coll p.1 s.1 p.2 s.2)) := by
      rw [ex_congr fun ξ => ex_const_mul c _, ex_const_mul,
        ex_prod (fun ξ₁ ξ₂ => Pr (fun s : Smp ξ₁ × Smp ξ₂ => coll ξ₁ s.1 ξ₂ s.2))]
      congr 1
      refine ex_congr fun ξ => ?_
      rw [ex_comm]
      exact ex_congr fun ξ₂ => pr_prod (fun (s : Smp ξ) (s₂ : Smp ξ₂) => coll ξ s ξ₂ s₂)
    rw [e1]
    calc _ ≤ Ex (fun ξ => Ex (fun s : Smp ξ => c * Ex (fun ξ₂ => Pr (coll ξ s ξ₂)))) :=
          Ex_mono fun ξ => Ex_mono fun s => hpt ξ s
      _ = _ := e2
      _ ≤ c * K := mul_le_mul_of_nonneg_left hK hc
  have hsq : ∑ u, a u ^ 2 ≤ ∑ u, a u * p u :=
    Finset.sum_le_sum fun u _ => by rw [sq]; exact mul_le_mul_of_nonneg_left (hap u) (ha0 u)
  have hcs := sq_sum_le_card_mul_sum_sq (s := (Finset.univ : Finset ι)) (f := a)
  rw [Finset.card_univ] at hcs
  calc Ex (fun ξ => Pr (fun s : Smp ξ => ∃ u, off u ξ s)) ≤ ∑ u, a u := h1
    _ = Real.sqrt ((∑ u, a u) ^ 2) := (Real.sqrt_sq (Finset.sum_nonneg fun u _ => ha0 u)).symm
    _ ≤ Real.sqrt (c * Fintype.card ι * K) := Real.sqrt_le_sqrt <| by
        calc (∑ u, a u) ^ 2 ≤ (Fintype.card ι : ℝ) * ∑ u, a u ^ 2 := hcs
          _ ≤ (Fintype.card ι : ℝ) * (c * K) := mul_le_mul_of_nonneg_left (hsq.trans h3) (Nat.cast_nonneg _)
          _ = c * Fintype.card ι * K := by ring

/-! ## A unit bound in a witness and unbound in a transcript -/

/-- **A unit unbound in `Xe` and bound in `Xw` gives a wire bit where they differ**: where `Xw` is correct at `u` and
opened at each of its wires, a unit unbound in `Xe` has a wire where `Xe` differs from `Xw`, at some bit. -/
theorem unbound_drawn {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
    (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) {Xw Xe : C.Trace} {u : Fin n}
    (hc : P.Correct Xw u) (ho : ∀ g ∈ P.io u, opens g (Xw g)) (hu : u ∈ Accounting.unbound P opens Xe) :
    ∃ g ∈ P.io u, ∃ i < C.w g, (Xw g).getLsbD i = !(Xe g).getLsbD i := by
  classical
  have hg : ∃ g ∈ P.io u, Xe g ≠ Xw g := by
    by_contra hall
    push_neg at hall
    have hu' := (Finset.mem_filter.1 hu).2
    rcases hu' with hw | ⟨g, hg, hng⟩
    · exact hw (P.correct_congr hall hc)
    · exact hng (by rw [hall g hg]; exact ho g hg)
  obtain ⟨g, hg, hne⟩ := hg
  refine ⟨g, hg, ?_⟩
  by_contra hall
  push_neg at hall
  refine hne (BitVec.eq_of_getLsbD_eq fun i hi => ?_)
  have := hall i hi
  cases h₁ : (Xe g).getLsbD i <;> cases h₂ : (Xw g).getLsbD i <;> simp_all

/-! ## The plays and their plurality -/

/-- **The plays the majority reads**, over a circuit `C` partitioned into `n` units by `P`: the outer samples `Ξ`, each
with a finite space of plays `Smp ξ`; at each play, whether it accepts (`acc`), the units it binds (`drawn`), the
witness the extractor returns (`wit`), the relation the witness must be in at the sample (`R`), and a witness read as
a transcript of `C` at the sample (`read`). -/
structure Plays (C : Verity.Circuit) (n : ℕ) (P : Verity.Partition C n) where
  Ξ : Type
  [finΞ : Fintype Ξ]
  Smp : Ξ → Type
  [finSmp : ∀ ξ, Fintype (Smp ξ)]
  Wit : Type
  acc : (ξ : Ξ) → Smp ξ → Prop
  drawn : (ξ : Ξ) → Smp ξ → Fin n → Prop
  wit : (ξ : Ξ) → Smp ξ → Option Wit
  R : Ξ → Wit → Prop
  read : Ξ → Wit → C.Trace

attribute [instance] Plays.finΞ Plays.finSmp

namespace Plays

variable {C : Verity.Circuit} {n : ℕ} {P : Verity.Partition C n} (M : Plays C n P)

/-- **The event a bit's weight counts**: at the play `s` of the sample `ξ`, it accepts, `g` is a wire of a unit it
binds, and the extractor returns a witness in the relation whose reading has `bit` at bit `i` of `g`. -/
def wEvent (g : Fin C.N) (i : ℕ) (bit : Bool) (ξ : M.Ξ) (s : M.Smp ξ) : Prop :=
  M.acc ξ s ∧ (∃ u, M.drawn ξ s u ∧ g ∈ P.io u) ∧
    ∃ wt, M.wit ξ s = some wt ∧ M.R ξ wt ∧ (M.read ξ wt g).getLsbD i = bit

/-- **The weight of bit `i` of gate `g` reading `bit`**: `wEvent`'s probability, averaged over the samples. -/
noncomputable def weight (g : Fin C.N) (i : ℕ) (bit : Bool) : ℝ :=
  Ex fun ξ : M.Ξ => Pr fun s : M.Smp ξ => M.wEvent g i bit ξ s

/-- **The plurality transcript**: each bit of each gate is the value the extractor's witnesses read there most often,
over the samples and the accepted plays that bind a unit holding the gate (`false` on a tie). -/
noncomputable def extract : C.Trace := fun g =>
  BitVec.ofNat (C.w g) (∑ i ∈ Finset.range (C.w g),
    if M.weight g i false < M.weight g i true then 2 ^ i else 0)

/-- **The event the binding term counts at the unit `u`**: at the play `s` of the sample `ξ`, it accepts, binds `u`,
and the extractor returns a witness in the relation whose reading differs from `Xt` at a bit of a wire of `u`. -/
def offEvent (Xt : C.Trace) (u : Fin n) (ξ : M.Ξ) (s : M.Smp ξ) : Prop :=
  M.acc ξ s ∧ M.drawn ξ s u ∧
    ∃ wt, M.wit ξ s = some wt ∧ M.R ξ wt ∧
      ∃ g ∈ P.io u, ∃ i < C.w g, (M.read ξ wt g).getLsbD i = !(Xt g).getLsbD i

/-- **A play's binding of the unit `u`**: at the play `s` of the sample `ξ`, it accepts, binds `u`, and the extractor
returns a witness in the relation. -/
def drawEvent (u : Fin n) (ξ : M.Ξ) (s : M.Smp ξ) : Prop :=
  M.acc ξ s ∧ M.drawn ξ s u ∧ ∃ wt, M.wit ξ s = some wt ∧ M.R ξ wt

/-- **The event the binding term counts at the wire bit `b`**: `wEvent` at bit `b.2` of the wire `b.1`, reading the
bit `extract` didn't pick there. -/
def bitEvent (b : Σ g : Fin C.N, Fin (C.w g)) (ξ : M.Ξ) (s : M.Smp ξ) : Prop :=
  M.wEvent b.1 b.2 (!(M.extract b.1).getLsbD b.2) ξ s

/-- **Two plays that read a bound wire differently collide**: wherever the plays `s₁` of `ξ₁` and `s₂` of `ξ₂` return
witnesses in the relation that both bind a unit holding the wire `g` and read `g` differently, `coll` holds of the
two. -/
def Binds (coll : (ξ₁ : M.Ξ) → M.Smp ξ₁ → (ξ₂ : M.Ξ) → M.Smp ξ₂ → Prop) : Prop :=
  ∀ ξ₁ (s₁ : M.Smp ξ₁) ξ₂ (s₂ : M.Smp ξ₂) w₁ w₂ (g : Fin C.N), M.wit ξ₁ s₁ = some w₁ → M.wit ξ₂ s₂ = some w₂ →
    M.R ξ₁ w₁ → M.R ξ₂ w₂ → (∃ u, M.drawn ξ₁ s₁ u ∧ g ∈ P.io u) → (∃ u, M.drawn ξ₂ s₂ u ∧ g ∈ P.io u) →
    M.read ξ₁ w₁ g ≠ M.read ξ₂ w₂ g → coll ξ₁ s₁ ξ₂ s₂

theorem weight_eq (g : Fin C.N) (i : ℕ) (bit : Bool) :
    M.weight g i bit = Ex fun ξ => Pr (M.wEvent g i bit ξ) := rfl

/-- **The extracted transcript's bit is the plurality's**: bit `i` of gate `g` of `extract` is set exactly where its
weight at `true` exceeds its weight at `false`. -/
theorem extract_bit (g : Fin C.N) {i : ℕ} (hi : i < C.w g) :
    (M.extract g).getLsbD i = true ↔ M.weight g i false < M.weight g i true :=
  getLsbD_ofNat_sum _ (fun j => M.weight g j false < M.weight g j true) hi

/-- **The plurality's minority weighs at most its majority**: at a bit `i` of a wire `g`, the weight of the bit
`extract` didn't pick is at most that of the bit it picked. -/
theorem minority_le_majority (g : Fin C.N) {i : ℕ} (hi : i < C.w g) :
    M.weight g i (!(M.extract g).getLsbD i) ≤ M.weight g i ((M.extract g).getLsbD i) := by
  have hx := M.extract_bit g hi
  cases hb : (M.extract g).getLsbD i
  · exact not_lt.1 fun h => by rw [hb] at hx; exact Bool.false_ne_true (hx.2 h)
  · rw [hb] at hx; exact (hx.1 rfl).le

/-- **A play off the plurality collides with every play on it**: where the play `s₁` of `ξ₁` returns a witness in the
relation whose reading, at a bit `i` of a wire `g` of a unit it binds, is not the bit `extract` picked, the weight of
the picked bit is at most the probability, over a second, independent sample and play, that the two collide. -/
theorem majority_le_coll (coll : (ξ₁ : M.Ξ) → M.Smp ξ₁ → (ξ₂ : M.Ξ) → M.Smp ξ₂ → Prop) (hcoll : M.Binds coll)
    {ξ₁ : M.Ξ} {s₁ : M.Smp ξ₁} {w₁ : M.Wit} {g : Fin C.N} {i : ℕ}
    (hd₁ : ∃ u, M.drawn ξ₁ s₁ u ∧ g ∈ P.io u) (he₁ : M.wit ξ₁ s₁ = some w₁) (hR₁ : M.R ξ₁ w₁)
    (hb₁ : (M.read ξ₁ w₁ g).getLsbD i = !(M.extract g).getLsbD i) :
    M.weight g i ((M.extract g).getLsbD i) ≤ Ex (fun ξ₂ => Pr (coll ξ₁ s₁ ξ₂)) := by
  rw [weight_eq]
  refine Ex_mono fun ξ₂ => Pr_mono fun s₂ hs₂ => ?_
  obtain ⟨-, hd₂, w₂, he₂, hR₂, hb₂⟩ := hs₂
  have hne : M.read ξ₁ w₁ g ≠ M.read ξ₂ w₂ g := fun h => by
    rw [h, hb₂] at hb₁
    cases hx : (M.extract g).getLsbD i <;> simp [hx] at hb₁
  exact hcoll ξ₁ s₁ ξ₂ s₂ w₁ w₂ g he₁ he₂ hR₁ hR₂ hd₁ hd₂ hne

/-- **The plurality's bit weighs at least half a bound unit**: at a bit `i` of a wire `g` of the unit `u`, the weight
of the bit `extract` picked there is at least half the probability that a play accepts, binds `u` and the extractor
returns a witness in the relation. -/
theorem drawn_le_weight {u : Fin n} {g : Fin C.N} (hg : g ∈ P.io u) {i : ℕ} (hi : i < C.w g) :
    Ex (fun ξ => Pr (M.drawEvent u ξ)) ≤ 2 * M.weight g i ((M.extract g).getLsbD i) := by
  have hsum : Ex (fun ξ => Pr (M.drawEvent u ξ)) ≤ M.weight g i true + M.weight g i false := by
    rw [weight_eq, weight_eq, ← Claims.Ex_add]
    refine Ex_mono fun ξ => le_trans (Pr_mono fun s hs => ?_) (pr_or_le _ _)
    obtain ⟨hacc, hu, wt, hw, hR⟩ := hs
    cases hb : (M.read ξ wt g).getLsbD i
    · exact Or.inr ⟨hacc, ⟨u, hu, hg⟩, wt, hw, hR, hb⟩
    · exact Or.inl ⟨hacc, ⟨u, hu, hg⟩, wt, hw, hR, hb⟩
  have hx := M.extract_bit g hi
  cases hb : (M.extract g).getLsbD i
  · have : M.weight g i true ≤ M.weight g i false :=
      not_lt.1 fun h => by rw [hb] at hx; exact Bool.false_ne_true (hx.2 h)
    linarith
  · have : M.weight g i false < M.weight g i true := by rw [hb] at hx; exact hx.1 rfl
    linarith

/-- **The binding term at the bits**: averaged over the samples, the probability that a play accepts while the
extractor's witness differs from the plurality at a bit of a wire of a unit it binds is at most `√(W · K)`, `W` the
circuit's wire bits and `K` the probability that two independent plays collide (`units_bound` over the bits). -/
theorem ex_off_le_bits (coll : (ξ₁ : M.Ξ) → M.Smp ξ₁ → (ξ₂ : M.Ξ) → M.Smp ξ₂ → Prop) (hcoll : M.Binds coll)
    {K : ℝ} (hK : Ex (fun p : M.Ξ × M.Ξ => Pr (fun s : M.Smp p.1 × M.Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤ K) :
    Ex (fun ξ => Pr (fun s : M.Smp ξ => ∃ u, M.offEvent M.extract u ξ s)) ≤
      Real.sqrt ((∑ g, C.w g : ℕ) * K) := by
  classical
  have hlow : ∀ b ξ₁ (s₁ : M.Smp ξ₁), M.bitEvent b ξ₁ s₁ →
      Ex (fun ξ => Pr (M.bitEvent b ξ)) ≤ 1 * Ex (fun ξ₂ => Pr (coll ξ₁ s₁ ξ₂)) := by
    intro b ξ₁ s₁ hs
    obtain ⟨-, hd₁, w₁, he₁, hR₁, hb₁⟩ := hs
    rw [one_mul, show Ex (fun ξ => Pr (M.bitEvent b ξ)) = M.weight b.1 b.2 (!(M.extract b.1).getLsbD b.2) from rfl]
    exact (M.minority_le_majority b.1 b.2.isLt).trans (M.majority_le_coll coll hcoll hd₁ he₁ hR₁ hb₁)
  have h := units_bound (Smp := M.Smp) M.bitEvent M.bitEvent coll zero_le_one (fun _ _ _ hs => hs) hlow hK
  simp only [Fintype.card_sigma, Fintype.card_fin, one_mul] at h
  refine le_trans (Ex_mono fun ξ => Pr_mono fun s hs => ?_) h
  obtain ⟨u, hacc, hu, wt, hw, hR, g, hg, i, hi, hbit⟩ := hs
  exact ⟨⟨g, i, hi⟩, hacc, ⟨u, hu, hg⟩, wt, hw, hR, hbit⟩

/-- **The binding term at the units**: averaged over the samples, the probability that a play accepts while the
extractor's witness differs from the plurality at a bit of a wire of a unit it binds is at most `√(2n · K)`, `n` the
units (`units_bound`): a play off the plurality at a bit of a wire of `u` collides with every play that reads the
plurality's bit there, which weighs at least half `u`'s binding (`drawn_le_weight`). -/
theorem ex_off_le_units (coll : (ξ₁ : M.Ξ) → M.Smp ξ₁ → (ξ₂ : M.Ξ) → M.Smp ξ₂ → Prop) (hcoll : M.Binds coll)
    {K : ℝ} (hK : Ex (fun p : M.Ξ × M.Ξ => Pr (fun s : M.Smp p.1 × M.Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤ K) :
    Ex (fun ξ => Pr (fun s : M.Smp ξ => ∃ u, M.offEvent M.extract u ξ s)) ≤ Real.sqrt (2 * n * K) := by
  classical
  have hlow : ∀ u ξ₁ (s₁ : M.Smp ξ₁), M.offEvent M.extract u ξ₁ s₁ →
      Ex (fun ξ => Pr (M.drawEvent u ξ)) ≤ 2 * Ex (fun ξ₂ => Pr (coll ξ₁ s₁ ξ₂)) := by
    intro u ξ₁ s₁ hs
    obtain ⟨-, hu₁, w₁, he₁, hR₁, g, hg, i, hi, hb₁⟩ := hs
    exact le_trans (M.drawn_le_weight hg hi) (mul_le_mul_of_nonneg_left
      (M.majority_le_coll coll hcoll ⟨u, hu₁, hg⟩ he₁ hR₁ hb₁) (by norm_num))
  have h := units_bound (Smp := M.Smp) (M.offEvent M.extract) M.drawEvent coll (by norm_num : (0 : ℝ) ≤ 2)
    (fun u ξ s hs => by
      obtain ⟨hacc, hu, wt, hw, hR, -⟩ := hs
      exact ⟨hacc, hu, wt, hw, hR⟩) hlow hK
  rwa [Fintype.card_fin] at h

/-- **The binding term**: averaged over the samples, the probability that a play accepts while the extractor's witness
differs from the plurality at a bit of a wire of a unit it binds is at most `√(min(W, 2n) · K)` (`ex_off_le_bits`,
`ex_off_le_units`). -/
theorem ex_off_le (coll : (ξ₁ : M.Ξ) → M.Smp ξ₁ → (ξ₂ : M.Ξ) → M.Smp ξ₂ → Prop) (hcoll : M.Binds coll)
    {K : ℝ} (hK : Ex (fun p : M.Ξ × M.Ξ => Pr (fun s : M.Smp p.1 × M.Smp p.2 => coll p.1 s.1 p.2 s.2)) ≤ K) :
    Ex (fun ξ => Pr (fun s : M.Smp ξ => ∃ u, M.offEvent M.extract u ξ s)) ≤
      Real.sqrt (min ((∑ g, C.w g : ℕ) : ℝ) (2 * n) * K) := by
  rcases le_total ((∑ g, C.w g : ℕ) : ℝ) (2 * n) with h | h
  · rw [min_eq_left h]; exact M.ex_off_le_bits coll hcoll hK
  · rw [min_eq_right h]; exact M.ex_off_le_units coll hcoll hK

end Plays

end Verity.Majority
