module

public import Proofs.Core.SamplingGeneric.FirstHit

/-! The first-hit finder by class, where a run opens a class whole or not at all (`hall`), without which the count by class fails. -/

@[expose] public section

open Finset

namespace Verity.FirstHit

open Verity.Rewinding Claims

variable {Ω P V Q : Type}

open Classical in
/-- **A class's table**: a run opens class `q` when it opens some node of the class. -/
noncomputable def clsOp (cls : P → Q) (op : Ω → P → Option V) (ω : Ω) (q : Q) : Option Unit :=
  if ∃ p, cls p = q ∧ (op ω p).isSome then some () else none

open Classical in
/-- **A class's sample**: a run's sample holds class `q` when it holds some node of the class. -/
noncomputable def clsChk (cls : P → Q) (chk : Ω → P → Bool) (ω : Ω) (q : Q) : Bool :=
  if ∃ p, cls p = q ∧ chk ω p = true then true else false

theorem clsChk_iff {cls : P → Q} {chk : Ω → P → Bool} {ω : Ω} {q : Q} :
    clsChk cls chk ω q = true ↔ ∃ p, cls p = q ∧ chk ω p = true := by
  rw [clsChk]
  by_cases hex : ∃ p, cls p = q ∧ chk ω p = true
  · rw [ite_eq_left hex]
    exact ⟨fun _ => hex, fun _ => rfl⟩
  · rw [ite_eq_right hex]
    exact ⟨fun h => absurd h Bool.false_ne_true, fun h => absurd h hex⟩

theorem clsOp_isSome {cls : P → Q} {op : Ω → P → Option V} {ω : Ω} {p : P} (h : (op ω p).isSome) :
    (clsOp cls op ω (cls p)).isSome := by
  have hex : ∃ p', cls p' = cls p ∧ (op ω p').isSome := ⟨p, rfl, h⟩
  rw [clsOp, ite_eq_left hex]
  rfl

theorem clsOp_isNone {cls : P → Q} {op : Ω → P → Option V}
    (hall : ∀ ω p p', cls p = cls p' → (op ω p).isSome → (op ω p').isSome) {ω : Ω} {p : P}
    (h : (op ω p).isNone) : (clsOp cls op ω (cls p)).isNone := by
  have hex : ¬ ∃ p', cls p' = cls p ∧ (op ω p').isSome := by
    rintro ⟨p', hp', hs⟩
    have h2 := hall ω p' p hp' hs
    cases hop : op ω p with
    | none => rw [hop] at h2; simp at h2
    | some v => rw [hop] at h; simp at h
  rw [clsOp, ite_eq_right hex]
  rfl

theorem clsChk_of {cls : P → Q} {chk : Ω → P → Bool} {ω : Ω} {p : P} (h : chk ω p = true) :
    clsChk cls chk ω (cls p) = true :=
  clsChk_iff.2 ⟨p, rfl, h⟩

theorem image_opened_subset [Fintype P] [Fintype Q] [DecidableEq Q] (cls : P → Q) (op : Ω → P → Option V) (ω : Ω) :
    (univ.filter fun p => (op ω p).isSome).image cls ⊆ univ.filter fun q => (clsOp cls op ω q).isSome := by
  intro q hq
  obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hq
  exact Finset.mem_filter.2 ⟨Finset.mem_univ _, clsOp_isSome (Finset.mem_filter.1 hp).2⟩

theorem filter_clsChk [Fintype P] [Fintype Q] [DecidableEq Q] (cls : P → Q) (chk : Ω → P → Bool) (ω : Ω) :
    (univ.filter fun q => clsChk cls chk ω q) = (univ.filter fun p => chk ω p).image cls := by
  ext q
  rw [Finset.mem_filter, Finset.mem_image, clsChk_iff]
  constructor
  · rintro ⟨_, p, hq, hc⟩
    exact ⟨p, Finset.mem_filter.2 ⟨Finset.mem_univ _, hc⟩, hq⟩
  · rintro ⟨p, hp, hq⟩
    exact ⟨Finset.mem_univ _, p, hq, (Finset.mem_filter.1 hp).2⟩

theorem loopCostAt_cls_le [DecidableEq Q] (cls : P → Q) (op : Ω → P → Option V) (chk : Ω → P → Bool)
    (cq : Ω → Q → ℕ) (hall : ∀ ω p p', cls p = cls p' → (op ω p).isSome → (op ω p').isSome) (K : ℕ) :
    ∀ (need : Finset P) (need' : Finset Q) (ρ : Fin K → Ω), need.image cls ⊆ need' →
      loopCostAt op chk (fun ω J => ∑ q ∈ J.image cls, cq ω q) K need ρ ≤
        loopCostAt (clsOp cls op) (clsChk cls chk) (fun ω J => ∑ q ∈ J, cq ω q) K need' ρ := by
  induction K with
  | zero =>
    intro need need' ρ _
    exact Nat.zero_le _
  | succ K ih =>
    intro need need' ρ hsub
    have e : loopCostAt op chk (fun ω J => ∑ q ∈ J.image cls, cq ω q) (K + 1) need ρ =
        ∑ q ∈ (need.filter fun p => chk (ρ 0) p).image cls, cq (ρ 0) q +
          loopCostAt op chk (fun ω J => ∑ q ∈ J.image cls, cq ω q) K
            (need.filter fun p => (op (ρ 0) p).isNone) (fun i => ρ i.succ) := rfl
    have e' : loopCostAt (clsOp cls op) (clsChk cls chk) (fun ω J => ∑ q ∈ J, cq ω q) (K + 1) need' ρ =
        ∑ q ∈ need'.filter (fun q => clsChk cls chk (ρ 0) q), cq (ρ 0) q +
          loopCostAt (clsOp cls op) (clsChk cls chk) (fun ω J => ∑ q ∈ J, cq ω q) K
            (need'.filter fun q => (clsOp cls op (ρ 0) q).isNone) (fun i => ρ i.succ) := rfl
    rw [e, e']
    refine Nat.add_le_add (Finset.sum_le_sum_of_subset ?_) (ih _ _ _ ?_)
    · intro q hq
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hq
      obtain ⟨hpn, hpc⟩ := Finset.mem_filter.1 hp
      exact Finset.mem_filter.2 ⟨hsub (Finset.mem_image_of_mem cls hpn), clsChk_of hpc⟩
    · intro q hq
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hq
      obtain ⟨hpn, hpo⟩ := Finset.mem_filter.1 hp
      exact Finset.mem_filter.2 ⟨hsub (Finset.mem_image_of_mem cls hpn), clsOp_isNone hall hpo⟩

theorem le_loopCostAt_cls [DecidableEq Q] (cls : P → Q) (op : Ω → P → Option V) (chk : Ω → P → Bool)
    (cq : Ω → Q → ℕ) (p : P) (hchk : ∀ ω, (op ω p).isSome → chk ω p = true) :
    ∀ (K : ℕ) (need : Finset P) (ρ : Fin K → Ω) (j : Fin K), p ∈ need → firstIdx op p K ρ = some j →
      cq (ρ j) (cls p) ≤ loopCostAt op chk (fun ω J => ∑ q ∈ J.image cls, cq ω q) K need ρ :=
  le_loopCostAt op chk (fun ω J => ∑ q ∈ J.image cls, cq ω q) p (fun ω => cq ω (cls p)) hchk fun ω _ hp =>
    Finset.single_le_sum (f := cq ω) (fun _ _ => Nat.zero_le _) (Finset.mem_image_of_mem cls hp)

theorem ex_costCls_tape_le_ex {T : Type} [Fintype T] [Fintype Ω] [Nonempty Ω] [Fintype P] [Fintype Q]
    [DecidableEq Q] (cls : P → Q) (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool)
    (hall : ∀ t ω p p', cls p = cls p' → (op t ω p).isSome → (op t ω p').isSome) (c : T → Ω → ℕ)
    (cq : T → Ω → Q → ℕ) (K : ℕ) :
    Ex (fun s : T × (Ω × (Fin K → Ω)) =>
        ((c s.1 s.2.1 + loopCostAt (op s.1) (chk s.1) (fun ω J => ∑ q ∈ J.image cls, cq s.1 ω q) K
          (univ.filter fun p => (op s.1 s.2.1 p).isSome) s.2.2 : ℕ) : ℝ)) ≤
      Ex (fun a : T × Ω =>
        ((c a.1 a.2 + ∑ q ∈ (univ.filter fun p : P => chk a.1 a.2 p).image cls, cq a.1 a.2 q : ℕ) : ℝ)) := by
  have h1 : ∀ s : T × (Ω × (Fin K → Ω)),
      ((c s.1 s.2.1 + loopCostAt (op s.1) (chk s.1) (fun ω J => ∑ q ∈ J.image cls, cq s.1 ω q) K
          (univ.filter fun p => (op s.1 s.2.1 p).isSome) s.2.2 : ℕ) : ℝ) ≤
        ((c s.1 s.2.1 + loopCostAt (clsOp cls (op s.1)) (clsChk cls (chk s.1)) (fun ω J => ∑ q ∈ J, cq s.1 ω q) K
          (univ.filter fun q => (clsOp cls (op s.1) s.2.1 q).isSome) s.2.2 : ℕ) : ℝ) := fun s =>
    Nat.cast_le.2 (Nat.add_le_add_left
      (loopCostAt_cls_le cls (op s.1) (chk s.1) (cq s.1) (hall s.1) K _ _ s.2.2
        (image_opened_subset cls (op s.1) s.2.1)) _)
  have h2 : Ex (fun s : T × (Ω × (Fin K → Ω)) =>
        ((c s.1 s.2.1 + loopCostAt (clsOp cls (op s.1)) (clsChk cls (chk s.1)) (fun ω J => ∑ q ∈ J, cq s.1 ω q) K
          (univ.filter fun q => (clsOp cls (op s.1) s.2.1 q).isSome) s.2.2 : ℕ) : ℝ)) ≤
      Ex (fun a : T × Ω =>
        ((c a.1 a.2 + ∑ q ∈ univ.filter (fun q : Q => clsChk cls (chk a.1) a.2 q), cq a.1 a.2 q : ℕ) : ℝ)) :=
    ex_costAt_tape_le_ex (fun t => clsOp cls (op t)) (fun t => clsChk cls (chk t)) c
      (fun t ω J => ∑ q ∈ J, cq t ω q) cq (fun _ _ _ => le_rfl) K
  have h3 : ∀ a : T × Ω,
      ((c a.1 a.2 + ∑ q ∈ univ.filter (fun q : Q => clsChk cls (chk a.1) a.2 q), cq a.1 a.2 q : ℕ) : ℝ) ≤
        ((c a.1 a.2 + ∑ q ∈ (univ.filter fun p : P => chk a.1 a.2 p).image cls, cq a.1 a.2 q : ℕ) : ℝ) := fun a => by
    rw [filter_clsChk]
  exact (Ex_mono h1).trans (h2.trans (Ex_mono h3))

theorem ex_cost_classFinderAt_le {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [Fintype Q] [DecidableEq Q] (cls : P → Q) (op : T → Ω → P → Option V) (chk : T → Ω → P → Bool)
    (hall : ∀ t ω p p', cls p = cls p' → (op t ω p).isSome → (op t ω p').isSome) (c : T → Ω → ℕ)
    (cq : T → Ω → Q → ℕ) (K : ℕ) (out : T × (Ω × (Fin K → Ω)) → Option (Msg × Msg)) :
    Ex (fun s => ((classFinderAt cls op chk c cq K out).cost s : ℝ)) ≤
      Ex (fun a : T × Ω =>
        ((c a.1 a.2 + ∑ q ∈ (univ.filter fun p : P => chk a.1 a.2 p).image cls, cq a.1 a.2 q : ℕ) : ℝ)) :=
  ex_costCls_tape_le_ex cls op chk hall c cq K

theorem pr_off_cls_le_finder {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P]
    [DecidableEq Q] [DecidableEq V] (H : Msg → Msg) (cls : P → Q) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool) (c : T → Ω → ℕ) (cq : T → Ω → Q → ℕ) (K : ℕ)
    (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v) :
    Pr (fun a : T × Ω => ∃ p, Off (op a.1) (vstar a.1) a.2 p) ≤
      2 * (classFinderAt cls op chk c cq K
          fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2).Pr H +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) :=
  pr_off_at_le_finder H op hv chk c (fun t ω J => ∑ q ∈ J.image cls, cq t ω q) K coll hc hs

/-- `hchk`, that a run opens only nodes of its sample, is a premise the proof doesn't read: under it the loop
charges what the output reads (`le_loopCostAt_cls`). -/
theorem pr_off_cls_le_ex_cap {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P] [Fintype Q]
    [DecidableEq Q] [DecidableEq V] (H : Msg → Msg) (cls : P → Q) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (_hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true)
    (hall : ∀ t ω p p', cls p = cls p' → (op t ω p).isSome → (op t ω p').isSome) (c : T → Ω → ℕ)
    (cq : T → Ω → Q → ℕ) (K : ℕ) (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v)
    (hCR : ExpectedCR H (classFinderAt cls op chk c cq K
      fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2)) :
    Pr (fun a : T × Ω => ∃ p, Off (op a.1) (vstar a.1) a.2 p) ≤
      2 * Ex (fun a : T × Ω =>
          ((c a.1 a.2 + ∑ q ∈ (univ.filter fun p : P => chk a.1 a.2 p).image cls, cq a.1 a.2 q : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) +
        (Fintype.card P : ℝ) / ((K : ℝ) + 1) := by
  have h1 := pr_off_cls_le_finder H cls op hv chk c cq K coll hc hs
  have h3 : (classFinderAt cls op chk c cq K
        fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2).Pr H ≤
      Ex (fun s => ((classFinderAt cls op chk c cq K
        fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2).cost s : ℝ)) /
        2 ^ (256 : ℝ) := hCR
  have h4 := div_le_div_of_nonneg_right
    (ex_cost_classFinderAt_le cls op chk hall c cq K
      fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2)
    (show (0 : ℝ) ≤ 2 ^ (256 : ℝ) by positivity)
  rw [mul_div_assoc]
  linarith

theorem pr_off_cls_le_ex {T : Type} [Fintype T] [Nonempty T] [Fintype Ω] [Nonempty Ω] [Fintype P] [Fintype Q]
    [DecidableEq Q] [DecidableEq V] (H : Msg → Msg) (cls : P → Q) (op : T → Ω → P → Option V) {vstar : T → P → V}
    (hv : ∀ t, IsPlurality (op t) (vstar t)) (chk : T → Ω → P → Bool)
    (hchk : ∀ (t : T) (ω : Ω) (p : P), (op t ω p).isSome → chk t ω p = true)
    (hall : ∀ t ω p p', cls p = cls p' → (op t ω p).isSome → (op t ω p').isSome) (c : T → Ω → ℕ)
    (cq : T → Ω → Q → ℕ) (coll : T → Ω → Ω → P → Option (Msg × Msg))
    (hc : ∀ (t : T) (ω ω' : Ω) (p : P), ConflictAt (op t) (ω, ω') p → (coll t ω ω' p).isSome)
    (hs : ∀ (t : T) (ω ω' : Ω) (p : P) (u v : Msg), coll t ω ω' p = some (u, v) → Collision H u v)
    (hCR : ∀ K : ℕ, ExpectedCR H (classFinderAt cls op chk c cq K
      fun s : T × (Ω × (Fin K → Ω)) => outOfAt (op s.1) (coll s.1) K s.2.1 s.2.2)) :
    Pr (fun a : T × Ω => ∃ p, Off (op a.1) (vstar a.1) a.2 p) ≤
      2 * Ex (fun a : T × Ω =>
          ((c a.1 a.2 + ∑ q ∈ (univ.filter fun p : P => chk a.1 a.2 p).image cls, cq a.1 a.2 q : ℕ) : ℝ)) /
        2 ^ (256 : ℝ) :=
  le_of_forall_cap fun K => pr_off_cls_le_ex_cap H cls op hv chk hchk hall c cq K coll hc hs (hCR K)

theorem key_clsOf {n : ℕ} {Key : Type} [DecidableEq Key] (key : Fin n → Key) (g : Fin n) :
    key (clsOf key g) = key g :=
  (mem_filter.1 (Finset.min'_mem (univ.filter fun g' => key g' = key g) _)).2

theorem clsOf_eq_iff {n : ℕ} {Key : Type} [DecidableEq Key] (key : Fin n → Key) (g g' : Fin n) :
    clsOf key g = clsOf key g' ↔ key g = key g' := by
  constructor
  · intro h
    exact (key_clsOf key g).symm.trans ((congrArg key h).trans (key_clsOf key g'))
  · intro h
    simp only [clsOf, h]

end Verity.FirstHit
