import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# The two-directional column game: from a winning holding to a schedule

`colRound` (`Pous.ColumnGame`) has four moves: forward lower, promotion (all columns available), key-gated
reverse (a whole top with whole top parents reveals its column of every lower label) and forward top.

`columnHard_of_schedule` reduces `ColumnHard G d n X` to a purely combinatorial **schedule bound**. A winning holding
yields a schedule:
* the dropped tops `U` with their first rounds `r` (increasing along edges);
* the lower labels whole by round `d − 1`, split into **held** `L0`, **promoted** `P` and **rebuilt** `R` (parents
  whole a round earlier), with first rounds `t`.

Counting each lower label's held columns:
* a rebuilt label saves the column of every dropped top placed after it;
* a promoted label saves nothing, and pays one held column for every held top whose column was not yet revealed
  (`ndHeld`);
* a held label pays `n − |U|`.
-/

namespace PousP3Column

open Pous Pous.ColumnGame

variable {n : ℕ}

/-! ### The game's invariants -/

section Game

variable (G : Labelling.TopoDAG n) (c0 : ColState n)

theorem cr_succ (r : ℕ) : colReach G c0 (r + 1) = colRound G (colReach G c0 r) := rfl

theorem low_mono {r r' : ℕ} (h : r ≤ r') :
    (colReach G c0 r).lowerWhole ⊆ (colReach G c0 r').lowerWhole := by
  induction h with
  | refl => exact Finset.Subset.refl _
  | step _ ih => exact ih.trans (by rw [cr_succ]; exact Finset.subset_union_left)

theorem top_mono {r r' : ℕ} (h : r ≤ r') :
    (colReach G c0 r).topWhole ⊆ (colReach G c0 r').topWhole := by
  induction h with
  | refl => exact Finset.Subset.refl _
  | step _ ih => exact ih.trans (by rw [cr_succ]; exact Finset.subset_union_left)

theorem mem_low_succ {r : ℕ} {w : Fin n} (h : w ∈ (colReach G c0 (r + 1)).lowerWhole) :
    w ∈ (colReach G c0 r).lowerWhole ∨ G.par w ⊆ (colReach G c0 r).lowerWhole ∨
      ∀ u, (w, u) ∈ (colReach G c0 r).lowerCol := by
  rw [cr_succ] at h
  simp only [colRound, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at h
  exact h

theorem mem_top_succ {r : ℕ} {u : Fin n} (h : u ∈ (colReach G c0 (r + 1)).topWhole) :
    u ∈ (colReach G c0 r).topWhole ∨
      ((∀ w, w ∈ (colReach G c0 r).lowerWhole ∨ (w, u) ∈ (colReach G c0 r).lowerCol) ∧
        G.par u ⊆ (colReach G c0 r).topWhole) := by
  rw [cr_succ] at h
  simp only [colRound, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and] at h
  exact h

/-- Every available column was held, or revealed by a whole top whose parents were whole. -/
theorem col_origin : ∀ (r : ℕ) (w v : Fin n), (w, v) ∈ (colReach G c0 r).lowerCol →
    (w, v) ∈ c0.lowerCol ∨ ∃ s < r, v ∈ (colReach G c0 s).topWhole ∧ G.par v ⊆ (colReach G c0 s).topWhole := by
  intro r
  induction r with
  | zero => intro w v h; exact Or.inl h
  | succ r ih =>
    intro w v h
    rw [cr_succ] at h
    simp only [colRound, Finset.mem_union, Finset.mem_product, Finset.mem_univ, true_and,
      Finset.mem_filter] at h
    rcases h with h | h
    · rcases ih w v h with h' | ⟨s, hs, h'⟩
      · exact Or.inl h'
      · exact Or.inr ⟨s, by omega, h'⟩
    · exact Or.inr ⟨r, by omega, h⟩

/-- The first round a top is whole. -/
noncomputable def ftop (u : Fin n) : ℕ :=
  open Classical in if h : ∃ r, u ∈ (colReach G c0 r).topWhole then Nat.find h else 0

theorem ftop_min {u : Fin n} {s : ℕ} (hs : u ∈ (colReach G c0 s).topWhole) : ftop G c0 u ≤ s := by
  classical
  have hex : ∃ r, u ∈ (colReach G c0 r).topWhole := ⟨s, hs⟩
  have he : ftop G c0 u = Nat.find hex := by unfold ftop; simp only [hex, ↓reduceDIte]
  rw [he]; exact Nat.find_min' hex hs

theorem ftop_mem {u : Fin n} {s : ℕ} (hs : u ∈ (colReach G c0 s).topWhole) :
    u ∈ (colReach G c0 (ftop G c0 u)).topWhole := by
  classical
  have hex : ∃ r, u ∈ (colReach G c0 r).topWhole := ⟨s, hs⟩
  have he : ftop G c0 u = Nat.find hex := by unfold ftop; simp only [hex, ↓reduceDIte]
  rw [he]; exact Nat.find_spec hex

/-- A dropped top placed by round `d` was placed at `ftop ≥ 1` from round `ftop − 1`. -/
theorem ftop_rule {u : Fin n} {d : ℕ} (hd : u ∈ (colReach G c0 d).topWhole) (h0 : u ∉ c0.topWhole) :
    1 ≤ ftop G c0 u ∧ ftop G c0 u ≤ d ∧
      (∀ w, w ∈ (colReach G c0 (ftop G c0 u - 1)).lowerWhole ∨ (w, u) ∈ (colReach G c0 (ftop G c0 u - 1)).lowerCol) ∧
      G.par u ⊆ (colReach G c0 (ftop G c0 u - 1)).topWhole := by
  have hmem := ftop_mem G c0 hd
  have hle := ftop_min G c0 hd
  have h1 : 1 ≤ ftop G c0 u := by
    rcases Nat.eq_zero_or_pos (ftop G c0 u) with h | h
    · rw [h] at hmem; exact absurd hmem h0
    · exact h
  obtain ⟨r, hr⟩ : ∃ r, ftop G c0 u = r + 1 := ⟨ftop G c0 u - 1, by omega⟩
  rw [hr] at hmem
  rcases mem_top_succ G c0 hmem with hprev | hrule
  · have := ftop_min G c0 hprev
    omega
  · refine ⟨h1, hle, ?_⟩
    rw [hr, Nat.add_sub_cancel]
    exact hrule

/-- The first round a lower label is whole. -/
noncomputable def flow (w : Fin n) : ℕ :=
  open Classical in if h : ∃ r, w ∈ (colReach G c0 r).lowerWhole then Nat.find h else 0

theorem flow_min {w : Fin n} {s : ℕ} (hs : w ∈ (colReach G c0 s).lowerWhole) : flow G c0 w ≤ s := by
  classical
  have hex : ∃ r, w ∈ (colReach G c0 r).lowerWhole := ⟨s, hs⟩
  have he : flow G c0 w = Nat.find hex := by unfold flow; simp only [hex, ↓reduceDIte]
  rw [he]; exact Nat.find_min' hex hs

theorem flow_mem {w : Fin n} {s : ℕ} (hs : w ∈ (colReach G c0 s).lowerWhole) :
    w ∈ (colReach G c0 (flow G c0 w)).lowerWhole := by
  classical
  have hex : ∃ r, w ∈ (colReach G c0 r).lowerWhole := ⟨s, hs⟩
  have he : flow G c0 w = Nat.find hex := by unfold flow; simp only [hex, ↓reduceDIte]
  rw [he]; exact Nat.find_spec hex

theorem flow_zero {w : Fin n} (h : w ∈ c0.lowerWhole) : flow G c0 w = 0 :=
  Nat.le_zero.1 (flow_min G c0 (s := 0) h)

/-- A lower label not held but whole by round `r` became whole at `flow ≥ 1`, by its parents or by promotion. -/
theorem flow_rule {w : Fin n} {r : ℕ} (hr : w ∈ (colReach G c0 r).lowerWhole) (h0 : w ∉ c0.lowerWhole) :
    1 ≤ flow G c0 w ∧ flow G c0 w ≤ r ∧
      (G.par w ⊆ (colReach G c0 (flow G c0 w - 1)).lowerWhole ∨
        ∀ u, (w, u) ∈ (colReach G c0 (flow G c0 w - 1)).lowerCol) := by
  have hmem := flow_mem G c0 hr
  have hle := flow_min G c0 hr
  have h1 : 1 ≤ flow G c0 w := by
    rcases Nat.eq_zero_or_pos (flow G c0 w) with h | h
    · rw [h] at hmem; exact absurd hmem h0
    · exact h
  obtain ⟨s, hs⟩ : ∃ s, flow G c0 w = s + 1 := ⟨flow G c0 w - 1, by omega⟩
  rw [hs] at hmem
  rcases mem_low_succ G c0 hmem with hprev | hrule
  · have := flow_min G c0 hprev
    omega
  · refine ⟨h1, hle, ?_⟩
    rw [hs, Nat.add_sub_cancel]
    exact hrule

end Game

/-! ### The schedule bound -/

/-- The held tops whose column is not yet revealed to a label promoted at round `t`: for every earlier round `s`
(`s + 2 ≤ t`), some parent is a dropped top not yet placed at `s`. -/
noncomputable def ndHeld (G : Labelling.TopoDAG n) (U : Finset (Fin n)) (r : Fin n → ℕ) (t : ℕ) :
    Finset (Fin n) :=
  open Classical in Finset.univ.filter fun v => v ∉ U ∧ ∀ s, s + 2 ≤ t → ∃ x ∈ G.par v, x ∈ U ∧ s < r x

/-- **Schedule bound.** For every schedule, the columns saved by rebuilt labels (`#{u ∈ U : t w < r u}` each) are
at most `X·n` plus what the held labels (`n − |U|` each) and the promoted labels (`|ndHeld|` each) pay. -/
def ScheduleBound (G : Labelling.TopoDAG n) (d X : ℕ) : Prop :=
  ∀ (U L0 P R : Finset (Fin n)) (r t : Fin n → ℕ),
    (∀ u ∈ U, 1 ≤ r u ∧ r u ≤ d) →
    (∀ a b, a ∈ G.par b → a ∈ U → b ∈ U → r a < r b) →
    (∀ w ∈ P, w ∉ L0 ∧ w ∉ R ∧ 1 ≤ t w) →
    (∀ w ∈ R, w ∉ L0 ∧ 1 ≤ t w ∧ ∀ a ∈ G.par w, (a ∈ L0 ∨ a ∈ P ∨ a ∈ R) ∧ t a < t w) →
    ∑ w ∈ R, (U.filter fun u => t w < r u).card ≤
      X * n + L0.card * (n - U.card) + ∑ p ∈ P, (ndHeld G U r (t p)).card

/-- **The reduction.** A schedule bound gives the column-saving certificate for the two-directional game. -/
theorem columnHard_of_schedule (G : Labelling.TopoDAG n) (d X : ℕ) (h : ScheduleBound G d X) :
    ColumnHard G d n X := by
  classical
  intro c0 hk
  have hall : ∀ v : Fin n, v ∈ (colReach G c0 d).topWhole := by
    have hsub : (colReach G c0 d).topWhole = Finset.univ :=
      Finset.eq_univ_of_card _ (le_antisymm (Finset.card_le_univ _) (by simpa using hk))
    intro v; rw [hsub]; exact Finset.mem_univ v
  set U := Finset.univ \ c0.topWhole with hUdef
  have hmemU : ∀ v, v ∈ U ↔ v ∉ c0.topWhole := by simp [hUdef]
  set r := ftop G c0 with hrdef
  set t := flow G c0 with htdef
  set L0 := c0.lowerWhole with hL0def
  set Ld := (colReach G c0 (d - 1)).lowerWhole with hLddef
  set R := (Ld \ L0).filter fun w => G.par w ⊆ (colReach G c0 (t w - 1)).lowerWhole with hRdef
  set P := (Ld \ L0).filter fun w => ¬ G.par w ⊆ (colReach G c0 (t w - 1)).lowerWhole with hPdef
  -- a dropped top is whole at round `s` iff placed by then
  have hTs : ∀ x ∈ U, ∀ s, x ∈ (colReach G c0 s).topWhole → r x ≤ s := fun x _ s hs => ftop_min G c0 hs
  have hrule : ∀ u ∈ U, 1 ≤ r u ∧ r u ≤ d ∧
      (∀ w, w ∈ (colReach G c0 (r u - 1)).lowerWhole ∨ (w, u) ∈ (colReach G c0 (r u - 1)).lowerCol) ∧
      G.par u ⊆ (colReach G c0 (r u - 1)).topWhole := fun u hu => ftop_rule G c0 (hall u) ((hmemU u).1 hu)
  have hfmin : ∀ w s, w ∈ (colReach G c0 s).lowerWhole → t w ≤ s := fun w s hs => flow_min G c0 hs
  have hfrule : ∀ w s, w ∈ (colReach G c0 s).lowerWhole → w ∉ L0 → 1 ≤ t w ∧ t w ≤ s ∧
      (G.par w ⊆ (colReach G c0 (t w - 1)).lowerWhole ∨ ∀ u, (w, u) ∈ (colReach G c0 (t w - 1)).lowerCol) :=
    fun w s hs h0 => flow_rule G c0 hs h0
  -- the column a dropped top needs from a lower label not yet whole is held
  have hneed : ∀ u ∈ U, ∀ w, w ∉ (colReach G c0 (r u - 1)).lowerWhole → (w, u) ∈ c0.lowerCol := by
    intro u hu w hw
    obtain ⟨h1, -, hcol, -⟩ := hrule u hu
    rcases hcol w with hw' | hw'
    · exact absurd hw' hw
    · rcases col_origin G c0 _ w u hw' with h' | ⟨s, hs, hsT, -⟩
      · exact h'
      · have := hTs u hu s hsT
        omega
  have hrU : ∀ u ∈ U, 1 ≤ r u ∧ r u ≤ d := fun u hu => ⟨(hrule u hu).1, (hrule u hu).2.1⟩
  have hrinc : ∀ a b, a ∈ G.par b → a ∈ U → b ∈ U → r a < r b := by
    intro a b hab ha hb
    obtain ⟨h1, -, -, hpar⟩ := hrule b hb
    have := hTs a ha _ (hpar hab)
    omega
  have hLdmem : ∀ w ∈ Ld \ L0, 1 ≤ t w ∧ t w ≤ d - 1 := fun w hw =>
    ⟨(hfrule w _ (Finset.mem_sdiff.1 hw).1 (Finset.mem_sdiff.1 hw).2).1,
      (hfrule w _ (Finset.mem_sdiff.1 hw).1 (Finset.mem_sdiff.1 hw).2).2.1⟩
  have hP : ∀ w ∈ P, w ∉ L0 ∧ w ∉ R ∧ 1 ≤ t w := by
    intro w hw
    have hw' := Finset.mem_filter.1 hw
    refine ⟨(Finset.mem_sdiff.1 hw'.1).2, fun hR => hw'.2 (Finset.mem_filter.1 hR).2, (hLdmem w hw'.1).1⟩
  have hR : ∀ w ∈ R, w ∉ L0 ∧ 1 ≤ t w ∧ ∀ a ∈ G.par w, (a ∈ L0 ∨ a ∈ P ∨ a ∈ R) ∧ t a < t w := by
    intro w hw
    have hw' := Finset.mem_filter.1 hw
    obtain ⟨ht1, htd⟩ := hLdmem w hw'.1
    refine ⟨(Finset.mem_sdiff.1 hw'.1).2, ht1, fun a ha => ?_⟩
    have haL := hw'.2 ha
    have hta : t a ≤ t w - 1 := hfmin a _ haL
    refine ⟨?_, by omega⟩
    by_cases h0 : a ∈ L0
    · exact Or.inl h0
    · have haLd : a ∈ Ld \ L0 := Finset.mem_sdiff.2 ⟨low_mono G c0 (by omega) haL, h0⟩
      by_cases hpa : G.par a ⊆ (colReach G c0 (t a - 1)).lowerWhole
      · exact Or.inr (Or.inr (Finset.mem_filter.2 ⟨haLd, hpa⟩))
      · exact Or.inr (Or.inl (Finset.mem_filter.2 ⟨haLd, hpa⟩))
  have hS := h U L0 P R r t hrU hrinc hP hR
  -- per lower label: held columns plus rebuilt savings cover the base
  set φw : Fin n → ℕ := fun w => if w ∈ R then (U.filter fun u => t w < r u).card else 0 with hφw
  set base : Fin n → ℕ := fun w =>
    if w ∈ L0 then 0 else if w ∈ P then U.card + (ndHeld G U r (t w)).card else U.card with hbase
  set fib : Fin n → ℕ := fun w => (c0.lowerCol.filter fun e => e.1 = w).card with hfib
  have hrow : ∀ w, base w ≤ fib w + φw w := by
    intro w
    simp only [hfib]
    by_cases hw0 : w ∈ L0
    · simp only [hbase, hw0, ↓reduceIte]; exact Nat.zero_le _
    · by_cases hwP : w ∈ P
      · have hwR : w ∉ R := (hP w hwP).2.1
        simp only [hbase, hφw, hw0, hwP, hwR, ↓reduceIte, add_zero]
        -- promoted: every column it needs for `U` and the unrevealed held tops is held
        have hwP' := Finset.mem_filter.1 hwP
        obtain ⟨ht1, -, hor⟩ := hfrule w _ (Finset.mem_sdiff.1 hwP'.1).1 hw0
        have hallc : ∀ u, (w, u) ∈ (colReach G c0 (t w - 1)).lowerCol := hor.resolve_left hwP'.2
        have hsub : (U ∪ ndHeld G U r (t w)).map (Function.Embedding.sectR w (Fin n)) ⊆
            c0.lowerCol.filter fun e => e.1 = w := by
          intro e he
          rw [Finset.mem_map] at he
          obtain ⟨v, hv, rfl⟩ := he
          refine Finset.mem_filter.2 ⟨?_, rfl⟩
          show (w, v) ∈ c0.lowerCol
          rcases Finset.mem_union.1 hv with hvU | hvN
          · by_cases hin : w ∈ (colReach G c0 (r v - 1)).lowerWhole
            · have htw : t w ≤ r v - 1 := hfmin w _ hin
              rcases col_origin G c0 _ w v (hallc v) with h' | ⟨s, hs, hsT, -⟩
              · exact h'
              · have := hTs v hvU s hsT
                omega
            · exact hneed v hvU w hin
          · rcases col_origin G c0 _ w v (hallc v) with h' | ⟨s, hs, hsT, hspar⟩
            · exact h'
            · exfalso
              have hvN' := (Finset.mem_filter.1 hvN).2
              obtain ⟨x, hx, hxU, hlt⟩ := hvN'.2 s (by omega)
              have := hTs x hxU s (hspar hx)
              omega
        have hcard := Finset.card_le_card hsub
        rw [Finset.card_map] at hcard
        have hdisj : Disjoint U (ndHeld G U r (t w)) :=
          Finset.disjoint_left.2 fun v hv hvN => (Finset.mem_filter.1 hvN).2.1 hv
        rw [Finset.card_union_of_disjoint hdisj] at hcard
        exact hcard
      · by_cases hwR : w ∈ R
        · simp only [hbase, hφw, hw0, hwP, hwR, ↓reduceIte]
          -- rebuilt: every dropped top placed no later than `t w` needs its column held
          have hsub : (U.filter fun u => ¬ t w < r u).map (Function.Embedding.sectR w (Fin n)) ⊆
              c0.lowerCol.filter fun e => e.1 = w := by
            intro e he
            rw [Finset.mem_map] at he
            obtain ⟨u, hu, rfl⟩ := he
            obtain ⟨huU, hlt⟩ := Finset.mem_filter.1 hu
            refine Finset.mem_filter.2 ⟨hneed u huU w fun hin => ?_, rfl⟩
            have := hfmin w _ hin
            have := (hrU u huU).1
            omega
          have hcard := Finset.card_le_card hsub
          rw [Finset.card_map] at hcard
          have hsplit := Finset.card_filter_add_card_filter_not (s := U) (fun u => t w < r u)
          omega
        · simp only [hbase, hφw, hw0, hwP, hwR, ↓reduceIte, add_zero]
          -- never whole by `d − 1`: every dropped top needs its column held
          have hwLd : w ∉ Ld := fun hin => by
            by_cases hpa : G.par w ⊆ (colReach G c0 (t w - 1)).lowerWhole
            · exact hwR (Finset.mem_filter.2 ⟨Finset.mem_sdiff.2 ⟨hin, hw0⟩, hpa⟩)
            · exact hwP (Finset.mem_filter.2 ⟨Finset.mem_sdiff.2 ⟨hin, hw0⟩, hpa⟩)
          have hsub : U.map (Function.Embedding.sectR w (Fin n)) ⊆ c0.lowerCol.filter fun e => e.1 = w := by
            intro e he
            rw [Finset.mem_map] at he
            obtain ⟨u, hu, rfl⟩ := he
            refine Finset.mem_filter.2 ⟨hneed u hu w fun hin => hwLd ?_, rfl⟩
            exact low_mono G c0 (by have := (hrU u hu).2; omega) hin
          have hcard := Finset.card_le_card hsub
          rwa [Finset.card_map] at hcard
  -- summing over lower labels
  have hsumrow : ∑ w, base w ≤ c0.lowerCol.card + ∑ w, φw w := by
    have hfibsum : c0.lowerCol.card = ∑ w, fib w := by
      rw [hfib]
      exact Finset.card_eq_sum_card_fiberwise (f := Prod.fst) (t := Finset.univ) fun _ _ => Finset.mem_univ _
    rw [hfibsum, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun w _ => hrow w
  have hφsum : ∑ w, φw w = ∑ w ∈ R, (U.filter fun u => t w < r u).card := by
    rw [hφw, Finset.sum_ite_mem, Finset.univ_inter]
  have hbasesum : ∑ w, base w = (n - L0.card) * U.card + ∑ p ∈ P, (ndHeld G U r (t p)).card := by
    have hPL0 : Disjoint P L0 := Finset.disjoint_left.2 fun w hw => (hP w hw).1
    have e1 : ∀ w, base w = (if w ∈ L0 then 0 else U.card) + (if w ∈ P then (ndHeld G U r (t w)).card else 0) := by
      intro w
      by_cases h0 : w ∈ L0
      · have : w ∉ P := fun hp => (hP w hp).1 h0
        simp [hbase, h0, this]
      · by_cases hp : w ∈ P <;> simp [hbase, h0, hp]
    rw [Finset.sum_congr rfl fun w _ => e1 w, Finset.sum_add_distrib, Finset.sum_ite_mem, Finset.univ_inter]
    congr 1
    rw [Finset.sum_ite, Finset.sum_const_zero, zero_add, Finset.sum_const, smul_eq_mul]
    congr 1
    rw [Finset.filter_not, Finset.card_sdiff_of_subset (Finset.filter_subset _ _), Finset.card_univ,
      Fintype.card_fin]
    simp
  have hT : c0.topWhole.card + U.card = n := by
    rw [hUdef, Finset.card_sdiff_of_subset (Finset.subset_univ _), Finset.card_univ, Fintype.card_fin]
    have := Finset.card_le_univ c0.topWhole
    rw [Fintype.card_fin] at this
    omega
  have hL0n : L0.card ≤ n := by have := Finset.card_le_univ L0; rwa [Fintype.card_fin] at this
  have hUn : U.card ≤ n := by omega
  rw [hφsum, hbasesum] at hsumrow
  unfold ColState.size
  show (n - X) * n ≤ n * L0.card + c0.lowerCol.card + n * c0.topWhole.card
  -- `(n − X)·n ≤ n·|L0| + |C0| + n·|T0|`
  have key : n * n ≤ n * L0.card + c0.lowerCol.card + n * c0.topWhole.card + X * n := by
    have e1 : (n - L0.card) * U.card + L0.card * U.card = n * U.card := by
      rw [← Nat.add_mul, Nat.sub_add_cancel hL0n]
    have e2 : L0.card * (n - U.card) + L0.card * U.card = L0.card * n := by
      rw [← Nat.mul_add, Nat.sub_add_cancel hUn]
    have e3 : n * c0.topWhole.card + n * U.card = n * n := by rw [← Nat.mul_add, hT]
    nlinarith [hsumrow, hS, e1, e2, e3]
  rcases le_or_gt X n with hX | hX
  · have : (n - X) * n + X * n = n * n := by rw [← Nat.add_mul, Nat.sub_add_cancel hX]
    omega
  · rw [Nat.sub_eq_zero_of_le hX.le, Nat.zero_mul]; exact Nat.zero_le _

end PousP3Column
