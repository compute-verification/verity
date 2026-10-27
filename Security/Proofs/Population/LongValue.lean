import Definitions.Population.LongValue
import Proofs.Population.Leakage

/-!
# Long values buy nothing

A prover may commit a value of any length, since an hm96 leaf's inner hash takes any length. The verifier's view is the
roots, the openings it accepts and its coins, and it refuses an opening of another length than the slot's width before
it reads it. So a long value is the prover's private memory plus one ordinary leaf: the slot it sits in has no opening
of its width (or the run holds a collision), the verifier's view of it is ⊥, and its writer and every reader are wrong.

* **The openings.** `view_congr`, `view_add_long`: the view reads only openings of the declared width, so a run that
  also commits long values has the view it would have without them. `view_of_accepts`, `long_value`: under
  `CollisionStep` (a named assumption: two strings that climb to one position give a collision among the run's hashed
  strings), an accepted opening is the view, and a slot a long value climbs to is ⊥, or the run holds a collision.
  `optCorrect_of_opened`, `opened_of_optCorrect`: a unit's proof holds exactly when the unit is correct in the view.
  `long_value_wrong`: the writer and every reader of a long value's slot are wrong.
* **⊥ is weakly dominated by a short wrong value.** `wrong_fill_subset`: putting any value of the slot's width in place
  of every ⊥ gives an assignment whose wrong units are among the ⊥ assignment's (`fill_mem_egressAt`: within the same
  budget, and showing the same values wherever the ⊥ assignment shows one). A unit that reads or writes ⊥ is wrong
  (`mem_optWrong_of_read_none`, `mem_optWrong_of_write_none`).
* **T1–T4 over `Option V`.** `liftFns f` applies `f` when every read holds a value and writes ⊥ otherwise; a unit
  correct over `Option V` is correct under it (`wrong_lift_subset`), so the population theorems hold over `Option V`
  with the wrong units of `optWrong`: robust determinacy and the cone (`opt_eq_of_writes`, `opt_eq_off_down`,
  `opt_eq_honest_off_down`), the egress count with `|V| + 1` for `|V|` (`card_optEgress_le`, `card_optEgressAt_le`,
  `card_optEgressAt_le_reach`), leakage beyond the honest egress (`opt_card_leak_le_coneEgress`,
  `opt_card_leak_le_reach`), and the skipped work (`opt_skipped_work_le_wt`, `opt_skipped_work_le_wt_of_luck`). The
  extra symbol is ⊥ itself, `log₂ (1 + 1/|V|)` bits a slot at most, and it costs its writer's and readers' correctness;
  read with ⊥ replaced by any short value, the counts are the population's with `|V|` (`fill_card_leak_le_coneEgress`).
-/

namespace Population

open Finset

namespace Dataflow

variable {U S : Type} (P : Dataflow U S)

open Classical in
/-- `f` over `Option V`: a unit applies `f` when every slot it reads holds a value, and writes ⊥ otherwise. -/
noncomputable def liftFns {V : Type} (f : P.Fns V) : P.Fns (Option V) := fun u r s =>
  if h : ∀ t, (r t).isSome then some (f u (fun t => (r t).get (h t)) s) else none

theorem liftFns_of_isSome {V : Type} (f : P.Fns V) {u : U} {r : P.reads u → Option V} (h : ∀ t, (r t).isSome)
    (s : P.writes u) : P.liftFns f u r s = some (f u (fun t => (r t).get (h t)) s) := by
  simp only [liftFns, dite_eq_left h]

/-- A unit correct over `Option V` is correct under `liftFns f`. -/
theorem correct_lift_of_optCorrect {V : Type} {f : P.Fns V} {σ : S → Option V} {u : U}
    (h : P.OptCorrect f σ u) : P.Correct (P.liftFns f) σ u := by
  obtain ⟨x, hr, hw⟩ := h
  have hs : ∀ t : P.reads u, ((fun r : P.reads u => σ r) t).isSome := fun t => by simp [hr t]
  have hx : (fun t : P.reads u => ((fun r : P.reads u => σ r) t).get (hs t)) = x := funext fun t => by simp [hr t]
  intro s
  rw [P.liftFns_of_isSome f hs s, hx, hw s]

/-- A unit whose reads all hold values and which is correct under `liftFns f` is correct over `Option V`. -/
theorem optCorrect_of_correct_lift {V : Type} {f : P.Fns V} {σ : S → Option V} {u : U}
    (hr : ∀ r : P.reads u, (σ r).isSome) (h : P.Correct (P.liftFns f) σ u) : P.OptCorrect f σ u := by
  refine ⟨fun r => (σ r).get (hr r), fun r => (Option.some_get (hr r)).symm, fun s => ?_⟩
  rw [h s, P.liftFns_of_isSome f hr s]

theorem wrong_lift_subset [Fintype U] {V : Type} {f : P.Fns V} {σ : S → Option V} :
    P.wrong (P.liftFns f) σ ⊆ P.optWrong f σ := by
  intro u hu
  simp only [wrong, optWrong, mem_filter, mem_univ, true_and] at hu ⊢
  exact fun hc => hu (P.correct_lift_of_optCorrect hc)

/-- Over assignments that hold a value everywhere, correctness over `Option V` is correctness. -/
theorem optCorrect_some_iff {V : Type} {f : P.Fns V} {τ : S → V} {u : U} :
    P.OptCorrect f (fun s => some (τ s)) u ↔ P.Correct f τ u := by
  constructor
  · rintro ⟨x, hr, hw⟩
    have hx : x = fun r : P.reads u => τ r := funext fun r => (Option.some_injective _ (hr r)).symm
    subst hx
    exact fun s => Option.some_injective _ (hw s)
  · exact fun h => ⟨fun r : P.reads u => τ r, fun r => rfl, fun s => congrArg some (h s)⟩

theorem optWrong_some [Fintype U] {V : Type} (f : P.Fns V) (τ : S → V) :
    P.optWrong f (fun s => some (τ s)) = P.wrong f τ := by
  ext u
  simp only [optWrong, wrong, mem_filter, mem_univ, true_and, P.optCorrect_some_iff]

/-- **The writer of ⊥ is wrong.** -/
theorem mem_optWrong_of_write_none [Fintype U] {V : Type} (f : P.Fns V) {σ : S → Option V} {u : U} {s : S}
    (hs : s ∈ P.writes u) (h : σ s = none) : u ∈ P.optWrong f σ := by
  simp only [optWrong, mem_filter, mem_univ, true_and]
  rintro ⟨x, -, hw⟩
  have := hw ⟨s, hs⟩
  simp [h] at this

/-- **Every reader of ⊥ is wrong.** -/
theorem mem_optWrong_of_read_none [Fintype U] {V : Type} (f : P.Fns V) {σ : S → Option V} {u : U} {r : S}
    (hr : r ∈ P.reads u) (h : σ r = none) : u ∈ P.optWrong f σ := by
  simp only [optWrong, mem_filter, mem_univ, true_and]
  rintro ⟨x, hx, -⟩
  have := hx ⟨r, hr⟩
  simp [h] at this

/-! ## ⊥ is weakly dominated by a short wrong value -/

theorem fill_of_some {V : Type} {d : S → V} {σ : S → Option V} {s : S} {v : V} (h : σ s = some v) :
    fill d σ s = v := by
  simp [fill, h]

/-- A unit correct over `Option V` is correct once ⊥ is filled with anything. -/
theorem correct_fill_of_optCorrect {V : Type} {f : P.Fns V} {σ : S → Option V} {u : U} (d : S → V)
    (h : P.OptCorrect f σ u) : P.Correct f (fill d σ) u := by
  obtain ⟨x, hr, hw⟩ := h
  have hx : (fun r : P.reads u => fill d σ r) = x := funext fun r => fill_of_some (hr r)
  intro s
  rw [hx]
  exact fill_of_some (hw s)

/-- **⊥ is weakly dominated by a short wrong value**: filling every ⊥ with any value of the slot's width wrongs no unit
that was correct. -/
theorem wrong_fill_subset [Fintype U] {V : Type} (f : P.Fns V) (σ : S → Option V) (d : S → V) :
    P.wrong f (fill d σ) ⊆ P.optWrong f σ := by
  intro u hu
  simp only [wrong, optWrong, mem_filter, mem_univ, true_and] at hu ⊢
  exact fun hc => hu (P.correct_fill_of_optCorrect d hc)

theorem holds_fill {V : Type} {a : S → V} {X : Finset S} {σ : S → Option V} (d : S → V) (h : P.OptHolds a X σ) :
    P.Holds a X (fill d σ) :=
  fun s hs hsX => fill_of_some (h s hs hsX)

/-- **Within the same budget**: an assignment over `Option V` with at most `m` wrong units, ⊥ filled with any short
values, is an assignment the population's count already covers. -/
theorem fill_mem_egressAt [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) {a : S → V}
    {X O : Finset S} {m : ℕ} {σ : S → Option V} (d : S → V) (hσa : P.OptHolds a X σ)
    (hσm : (P.optWrong f σ).card ≤ m) : (fun o : O => fill d σ o) ∈ P.egressAt f a X O m := by
  simp only [egressAt, mem_image, mem_filter, mem_univ, true_and]
  exact ⟨fill d σ, ⟨P.holds_fill d hσa, (card_le_card (P.wrong_fill_subset f σ d)).trans hσm⟩, rfl⟩

/-! ## T1: robust determinacy and the cone over `Option V` -/

/-- **Robust determinacy over `Option V`**: an assignment is determined by its inputs and the writes of a set
containing its wrong units, ⊥ included. -/
theorem opt_eq_of_writes [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {B : Finset U}
    {σ τ : S → Option V} (hin : ∀ s ∈ P.input, σ s = τ s) (hσ : P.optWrong f σ ⊆ B) (hτ : P.optWrong f τ ⊆ B)
    (hB : ∀ u ∈ B, ∀ s ∈ P.writes u, σ s = τ s) : σ = τ :=
  P.eq_of_writes hc ha (P.liftFns f) hin (P.wrong_lift_subset.trans hσ) (P.wrong_lift_subset.trans hτ) hB

/-- **The cone over `Option V`**. -/
theorem opt_eq_off_down [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {B : Finset U}
    {σ τ : S → Option V} (hin : ∀ s ∈ P.input, σ s = τ s) (hσ : P.optWrong f σ ⊆ B) (hτ : P.optWrong f τ ⊆ B) :
    ∀ v, v ∉ P.down B → ∀ s ∈ P.writes v, σ s = τ s :=
  P.eq_off_down hc ha (P.liftFns f) hin (P.wrong_lift_subset.trans hσ) (P.wrong_lift_subset.trans hτ)

/-- **Integrity against the honest run over `Option V`**: off the downstream cone of the wrong units, every slot holds
its honest value, never ⊥. -/
theorem opt_eq_honest_off_down [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest : S → V} {σ : S → Option V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = some (a s)) :
    ∀ v, v ∉ P.down (P.optWrong f σ) → ∀ s ∈ P.writes v, σ s = some (honest s) :=
  P.eq_off_down (P.covered_of_singleAssignment h.1) h.2 (P.liftFns f) (τ := fun s => some (honest s))
    (fun s hs => by rw [hσ s hs, hh s hs]) P.wrong_lift_subset
    (P.wrong_lift_subset.trans (by rw [P.optWrong_some, P.wrong_eq_empty hc]; exact empty_subset _))

/-! ## T2: the egress count over `Option V` -/

theorem optEgress_subset [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (B : Finset U) :
    P.optEgress f a X O B ⊆ P.egress (P.liftFns f) (fun s => some (a s)) X O B := by
  intro e he
  simp only [optEgress, egress, mem_image, mem_filter, mem_univ, true_and] at he ⊢
  obtain ⟨σ, ⟨hσa, hσw⟩, rfl⟩ := he
  exact ⟨σ, ⟨hσa, P.wrong_lift_subset.trans hσw⟩, rfl⟩

theorem optEgressAt_subset [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (m : ℕ) :
    P.optEgressAt f a X O m ⊆ P.egressAt (P.liftFns f) (fun s => some (a s)) X O m := by
  intro e he
  simp only [optEgressAt, egressAt, mem_image, mem_filter, mem_univ, true_and] at he ⊢
  obtain ⟨σ, ⟨hσa, hσm⟩, rfl⟩ := he
  exact ⟨σ, ⟨hσa, (card_le_card P.wrong_lift_subset).trans hσm⟩, rfl⟩

/-- **The capacity at a wrong set over `Option V`**: at most `(|V| + 1) ^ |writes B|` egress values. -/
theorem card_optEgress_le [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) {X O : Finset S} {B : Finset U} (hX : ∀ s ∈ X, s ∉ P.up O) :
    (P.optEgress f a X O B).card ≤ (Fintype.card V + 1) ^ (P.writesOf B).card := by
  rw [← Fintype.card_option]
  exact (card_le_card (P.optEgress_subset f a X O B)).trans (P.card_egress_le hc ha (P.liftFns f) _ hX)

/-- **The capacity with at most `m` wrong units over `Option V`**. -/
theorem card_optEgressAt_le [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) {X O : Finset S} (hX : ∀ s ∈ X, s ∉ P.up O) (m w : ℕ)
    (hw : ∀ u, (P.writes u).card ≤ w) :
    (P.optEgressAt f a X O m).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * (Fintype.card V + 1) ^ (m * w) := by
  rw [← Fintype.card_option]
  exact (card_le_card (P.optEgressAt_subset f a X O m)).trans (P.card_egressAt_le hc ha (P.liftFns f) _ hX m w hw)

/-- **Control reach over `Option V`**: with the inputs fixed, at most
`(∑_{j ≤ m} C(|U|, j)) · (|V| + 1) ^ min(m w, R)` egress values. -/
theorem card_optEgressAt_le_reach [Fintype U] [Fintype S] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    (f : P.Fns V) (a : S → V) (O : Finset S) (m w R : ℕ) (hw : ∀ u, (P.writes u).card ≤ w)
    (hR : ∀ B : Finset U, B.card ≤ m → (P.coneEgress O B).card ≤ R) :
    (P.optEgressAt f a ∅ O m).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * (Fintype.card V + 1) ^ min (m * w) R := by
  rw [← Fintype.card_option]
  exact (card_le_card (P.optEgressAt_subset f a ∅ O m)).trans
    (P.card_egressAt_le_reach hc ha (P.liftFns f) _ O m w R hw hR)

/-! ## T3: leakage beyond the honest egress over `Option V` -/

theorem card_image_some_le {ι : Type} {O : Finset S} {V : Type} [Fintype V] [DecidableEq V] (Cl : Finset ι)
    (honest : ι → S → V) :
    (Cl.image fun i (o : O) => some (honest i o)).card ≤ (Cl.image fun i (o : O) => honest i o).card := by
  have : (Cl.image fun i (o : O) => some (honest i o)) =
      (Cl.image fun i (o : O) => honest i o).image fun e o => some (e o) := by
    rw [image_image]
    rfl
  rw [this]
  exact card_image_le

/-- **The uniform form over `Option V`**: with at most `m` wrong units and every unit's cone egress `E(u) ≤ E`, the
egress takes at most `#honest · C(|U|, m) · (|V| + 1) ^ (m E)` values. -/
theorem opt_card_leak_le_coneEgress [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [DecidableEq V] (f : P.Fns V) (O : Finset S) {m E : ℕ} (hmU : m ≤ Fintype.card U)
    (hE : ∀ u, (P.coneEgress O {u}).card ≤ E) {ι : Type} (Cl : Finset ι) (σ : ι → S → Option V)
    (honest : ι → S → V) (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = some (honest i t))
    (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u) (hm : ∀ i ∈ Cl, (P.optWrong f (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * (Fintype.card U).choose m * (Fintype.card V + 1) ^ (m * E) := by
  have h := P.card_leak_le_coneEgress hc ha (P.liftFns f) O hmU hE Cl σ (fun i t => some (honest i t)) hin
    (fun i hi u => P.correct_lift_of_optCorrect (P.optCorrect_some_iff.2 (hh i hi u)))
    (fun i hi => (card_le_card P.wrong_lift_subset).trans (hm i hi))
  rw [Fintype.card_option] at h
  exact h.trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (card_image_some_le Cl honest)))

/-- **Leakage about the secrets is at most influence reach, over `Option V`**: at most
`#honest · (∑_{j ≤ m} C(|U|, j)) · (|V| + 1) ^ R` egress values. -/
theorem opt_card_leak_le_reach [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V] [DecidableEq V]
    (f : P.Fns V) (O : Finset S) (m R : ℕ) (hR : ∀ B : Finset U, B.card ≤ m → (P.coneEgress O B).card ≤ R)
    {ι : Type} (Cl : Finset ι) (σ : ι → S → Option V) (honest : ι → S → V)
    (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = some (honest i t)) (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u)
    (hm : ∀ i ∈ Cl, (P.optWrong f (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤ (Cl.image fun i (o : O) => honest i o).card *
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * (Fintype.card V + 1) ^ R := by
  have h := P.card_leak_le_reach hc ha (P.liftFns f) O m R hR Cl σ (fun i t => some (honest i t)) hin
    (fun i hi u => P.correct_lift_of_optCorrect (P.optCorrect_some_iff.2 (hh i hi u)))
    (fun i hi => (card_le_card P.wrong_lift_subset).trans (hm i hi))
  rw [Fintype.card_option] at h
  exact h.trans (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ (card_image_some_le Cl honest)))

/-- **Read with ⊥ filled, the count is the population's**: whatever short values stand in for ⊥, the egress takes at
most `#honest · C(|U|, m) · |V| ^ (m E)` values. -/
theorem fill_card_leak_le_coneEgress [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} [Fintype V]
    [DecidableEq V] [Nonempty V] (f : P.Fns V) (O : Finset S) {m E : ℕ} (hmU : m ≤ Fintype.card U)
    (hE : ∀ u, (P.coneEgress O {u}).card ≤ E) {ι : Type} (Cl : Finset ι) (σ : ι → S → Option V)
    (d honest : ι → S → V) (hin : ∀ i ∈ Cl, ∀ t ∈ P.input, σ i t = some (honest i t))
    (hh : ∀ i ∈ Cl, ∀ u, P.Correct f (honest i) u) (hm : ∀ i ∈ Cl, (P.optWrong f (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => fill (d i) (σ i) o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * (Fintype.card U).choose m * Fintype.card V ^ (m * E) :=
  P.card_leak_le_coneEgress hc ha f O hmU hE Cl (fun i => fill (d i) (σ i)) honest
    (fun i hi t ht => fill_of_some (hin i hi t ht)) hh
    (fun i hi => (card_le_card (P.wrong_fill_subset f (σ i) (d i))).trans (hm i hi))

/-! ## T4: the skipped work over `Option V` -/

theorem skipsLucky_lift {V : Type} {f : P.Fns V} {honest : S → V} {σ : S → Option V} {K L : Finset U}
    (hl : P.OptSkipsLucky f honest σ K L) : P.SkipsLucky (P.liftFns f) (fun s => some (honest s)) σ K L :=
  fun u hu hr hc => hl u hu hr
    (P.optCorrect_of_correct_lift (fun r => by rw [hr r.1 r.2]; rfl) hc)

/-- **The skipped work over `Option V`**: at most `F · wt(wrong)` plus the lucky work, the wrong units those of
`optWrong`, the readers and writers of ⊥ among them. -/
theorem opt_skipped_work_le_wt [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V)
    {honest : S → V} {σ : S → Option V} (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u)
    (hσ : ∀ s ∈ P.input, σ s = some (a s)) {K L : Finset U} (hl : P.OptSkipsLucky f honest σ K L) (wt : U → ℕ)
    (F : ℕ) (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F * wt u) :
    ∑ u ∈ univ \ K, wt u ≤ F * ∑ u ∈ P.optWrong f σ, wt u + ∑ u ∈ L, wt u :=
  (P.skipped_work_le_wt h (P.liftFns f) (fun s => some (a s)) (honest := fun s => some (honest s))
    (fun s hs => by rw [hh s hs]) (fun u => P.correct_lift_of_optCorrect (P.optCorrect_some_iff.2 (hc u))) hσ
    (P.skipsLucky_lift hl) wt F hF).trans
    (Nat.add_le_add_right (Nat.mul_le_mul_left _ (sum_le_sum_of_subset P.wrong_lift_subset)) _)

/-- **The skipped work under weighted sampling and a luck rate, over `Option V`**:
`(den − num) · W ≤ den · (F M + slack)`. -/
theorem opt_skipped_work_le_wt_of_luck [Fintype U] [DecidableEq U] (h : P.Checked) {V : Type} (f : P.Fns V)
    (a : S → V) {honest : S → V} {σ : S → Option V} (hh : ∀ s ∈ P.input, honest s = a s)
    (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = some (a s)) {K L : Finset U}
    (hl : P.OptSkipsLucky f honest σ K L) (wt : U → ℕ) (F : ℕ) (hF : ∀ u, ∑ v ∈ P.down {u}, wt v ≤ F * wt u)
    (M : ℕ) (hM : ∑ u ∈ P.optWrong f σ, wt u ≤ M) (num den slack : ℕ)
    (hluck : den * ∑ u ∈ L, wt u ≤ num * ∑ u ∈ univ \ K, wt u + den * slack) :
    (den - num) * ∑ u ∈ univ \ K, wt u ≤ den * (F * M + slack) := by
  have h1 := P.opt_skipped_work_le_wt h f a hh hc hσ hl wt F hF
  have h2 : F * ∑ u ∈ P.optWrong f σ, wt u ≤ F * M := Nat.mul_le_mul_left _ hM
  have h3 : den * ∑ u ∈ univ \ K, wt u ≤ den * (F * M) + den * ∑ u ∈ L, wt u := by
    rw [← Nat.mul_add]
    exact Nat.mul_le_mul_left _ (h1.trans (Nat.add_le_add_right h2 _))
  rw [Nat.sub_mul, Nat.mul_add]
  omega

end Dataflow

/-! ## The openings: a long value is ⊥ -/

namespace Openings

variable {S V : Type} (O : Openings S V)

theorem view_eq_none_iff {s : S} : O.view s = none ↔ ∀ x, ¬ O.Accepts s x := by
  by_cases h : ∃ x, O.Accepts s x
  · simp only [view, dite_eq_left h, reduceCtorEq, false_iff, not_forall, not_not]
    exact h
  · simp only [view, dite_eq_right h, true_iff]
    exact fun x hx => h ⟨x, hx⟩

theorem exists_of_view_eq_some {s : S} {v : V} (h : O.view s = some v) : ∃ x, O.Accepts s x ∧ O.read s x = v := by
  by_cases he : ∃ x, O.Accepts s x
  · rw [view, dite_eq_left he] at h
    exact ⟨he.choose, he.choose_spec, Option.some_injective _ h⟩
  · rw [view, dite_eq_right he] at h
    cases h

/-- **An accepted opening is the view**, or the run holds a collision. -/
theorem view_of_accepts {D : Type} {H : List UInt8 → D} (hcs : CollisionStep H O) {s : S} {x : List UInt8}
    (hx : O.Accepts s x) : O.view s = some (O.read s x) ∨ Collides H O.hashed := by
  have he : ∃ y, O.Accepts s y := ⟨x, hx⟩
  by_cases hxy : he.choose = x
  · left
    rw [view, dite_eq_left he, hxy]
  · exact Or.inr (hcs s _ he.choose_spec.2 x hx.2 hxy)

/-- **A long value leaves its slot ⊥**: when a string whose length is not the slot's width climbs to the slot's
position, the verifier's view of the slot is ⊥, or the run holds a collision. -/
theorem long_value {D : Type} {H : List UInt8 → D} (hcs : CollisionStep H O) {s : S} {x : List UInt8}
    (hx : x ∈ O.climbs s) (hlen : x.length ≠ O.width s) : O.view s = none ∨ Collides H O.hashed := by
  by_cases he : ∃ y, O.Accepts s y
  · obtain ⟨y, hyl, hy⟩ := he
    exact Or.inr (hcs s x hx y hy fun e => hlen ((congrArg List.length e).trans hyl))
  · exact Or.inl (O.view_eq_none_iff.2 fun y hy => he ⟨y, hy⟩)

/-- **The view reads only openings of the declared width**: two runs with the same widths and decoding, whose climbing
strings of the declared width are the same, have the same view, whatever strings of other lengths climb in either and
whatever they hash. -/
theorem view_congr {O' : Openings S V} (hw : O.width = O'.width) (hr : O.read = O'.read)
    (hc : ∀ s x, x.length = O.width s → (x ∈ O.climbs s ↔ x ∈ O'.climbs s)) : O.view = O'.view := by
  have hA : O.Accepts = O'.Accepts := by
    funext s x
    apply propext
    unfold Accepts
    rw [← hw]
    exact ⟨fun h => ⟨h.1, (hc s x h.1).1 h.2⟩, fun h => ⟨h.1, (hc s x h.1).2 h.2⟩⟩
  funext s
  unfold view
  rw [hA, hr]

/-- **Long values do not reach the view**: committing long values as well, and hashing anything more, leaves the
verifier's view as it was. -/
theorem view_add_long (L : S → Set (List UInt8)) (hL : ∀ s, ∀ x ∈ L s, x.length ≠ O.width s)
    (T : Set (List UInt8)) : ({ O with climbs := fun s => O.climbs s ∪ L s, hashed := T } : Openings S V).view =
      O.view := by
  exact Openings.view_congr (O := { O with climbs := fun s => O.climbs s ∪ L s, hashed := T }) (O' := O) rfl rfl
    fun s x hx => ⟨fun h => h.resolve_right fun hxL => hL s x hxL hx, fun h => Or.inl h⟩

end Openings

namespace Dataflow

variable {U S V : Type} (P : Dataflow U S) (O : Openings S V)

/-- **A unit's proof makes it correct in the view**, or the run holds a collision. -/
theorem optCorrect_of_opened {D : Type} {H : List UInt8 → D} (hcs : CollisionStep H O) {f : P.Fns V} {u : U}
    (h : P.Opened O f u) : P.OptCorrect f O.view u ∨ Collides H O.hashed := by
  by_cases hcol : Collides H O.hashed
  · exact Or.inr hcol
  · left
    obtain ⟨x, hr, hw⟩ := h
    refine ⟨x, fun r => ?_, fun s => ?_⟩
    · obtain ⟨y, hy, hyv⟩ := hr r
      rw [← hyv]
      exact (O.view_of_accepts hcs hy).resolve_right hcol
    · obtain ⟨y, hy, hyv⟩ := hw s
      rw [← hyv]
      exact (O.view_of_accepts hcs hy).resolve_right hcol

/-- **A unit correct in the view has a proof to give**: accepted openings of every slot it reads and writes. -/
theorem opened_of_optCorrect {f : P.Fns V} {u : U} (h : P.OptCorrect f O.view u) : P.Opened O f u := by
  obtain ⟨x, hr, hw⟩ := h
  exact ⟨x, fun r => O.exists_of_view_eq_some (hr r), fun s => O.exists_of_view_eq_some (hw s)⟩

/-- **The long-value lemma**: when a string whose length is not slot `s`'s width climbs to `s`'s position, `s`'s writer
and every unit that reads `s` are wrong in the verifier's view, or the run holds a collision. With
`wrong_fill_subset`, the attack is weakly dominated by writing a short wrong value at `s`. -/
theorem long_value_wrong [Fintype U] {D : Type} {H : List UInt8 → D} (hcs : CollisionStep H O) (f : P.Fns V)
    {s : S} {x : List UInt8} (hx : x ∈ O.climbs s) (hlen : x.length ≠ O.width s) :
    ((∀ u, s ∈ P.writes u → u ∈ P.optWrong f O.view) ∧ ∀ u, s ∈ P.reads u → u ∈ P.optWrong f O.view) ∨
      Collides H O.hashed :=
  (O.long_value hcs hx hlen).imp_left fun h =>
    ⟨fun _ hu => P.mem_optWrong_of_write_none f hu h, fun _ hu => P.mem_optWrong_of_read_none f hu h⟩

end Dataflow

end Population
