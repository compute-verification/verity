module

public import Definitions.Claims.Sampling

/-!
# What a proof block's correctness reads

`Prog.Correct ref u` reads `ref` only at `u`'s gates and the gates they read (`Prog.correct_congr`), and more finely
only where `Prog.Needs` says: at a gate of `u` whose value `u` commits, and at a gate outside `u`, other than a
constant, that one of `u`'s gates reads (`Prog.correct_of_needs`). `Prog.runIn` reads the inputs and the outside reads
at and before the gate it computes (`Prog.runIn_congr`). At a committed or outside gate `refIn` is the committed value
(`Prog.refIn_of_boundary`, `Prog.refIn_of_not_mem`), and an input, a returned gate and a gate read from outside are on
the boundary (`Prog.boundary_of_input`, `Prog.boundary_of_out`, `Prog.boundary_of_read`).
-/

@[expose] public section

namespace Claims.Sampling

/-! ## A proof block reads its gates and what they read -/

theorem mapM_rd_congr {rd₁ rd₂ : ℕ → Option Bool} :
    ∀ {ws : List ℕ}, (∀ i ∈ ws, rd₁ i = rd₂ i) → ws.mapM rd₁ = ws.mapM rd₂
  | [], _ => rfl
  | i :: ws, h => by
    rw [List.mapM_cons, List.mapM_cons, h i (by simp), mapM_rd_congr fun j hj => h j (by simp [hj])]

theorem BGate.val_congr {x : BGate} {rd₁ rd₂ : ℕ → Option Bool} {o₁ o₂ : Option Bool}
    (hrd : ∀ i ∈ x.reads, rd₁ i = rd₂ i) (ho : x = .input → o₁ = o₂) : x.val rd₁ o₁ = x.val rd₂ o₂ := by
  cases x with
  | input => exact ho rfl
  | const b => rfl
  | not i => simp only [BGate.val, hrd i (by simp [BGate.reads])]
  | and ws =>
    have h : ∀ i ∈ ws, rd₁ i = rd₂ i := hrd
    simp only [BGate.val, mapM_rd_congr h]
  | xor ws =>
    have h : ∀ i ∈ ws, rd₁ i = rd₂ i := hrd
    simp only [BGate.val, mapM_rd_congr h]

theorem Prog.read_congr (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {i : ℕ} (h : ref₁ i = ref₂ i) :
    C.read ref₁ i = C.read ref₂ i := by
  unfold Prog.read
  split
  · rfl
  · exact h

theorem Prog.read_of_not_const (C : Prog) (ref : ℕ → Option Bool) {i : ℕ} (h : ∀ b, C.gates[i]? ≠ some (.const b)) :
    C.read ref i = ref i := by
  unfold Prog.read
  split
  · rename_i b hb
    exact absurd hb (h b)
  · rfl

/-- **`runIn` reads the inputs and outside reads at and before `g`.** -/
theorem Prog.runIn_congr (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ} :
    ∀ g ∈ u, (∀ g' ∈ u, g' ≤ g → C.gates[g']? = some .input → ref₁ g' = ref₂ g') →
      (∀ g' ∈ u, g' ≤ g → ∀ x : BGate, C.gates[g']? = some x → ∀ i ∈ x.reads, i ∉ u →
        C.read ref₁ i = C.read ref₂ i) →
      C.runIn ref₁ u g = C.runIn ref₂ u g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg hin hrd
  rw [Prog.runIn, Prog.runIn]
  cases hx : C.gates[g]? with
  | none => rfl
  | some x =>
    refine BGate.val_congr (fun i hi => ?_) fun h => hin g hg le_rfl (hx.trans (congrArg some h))
    by_cases hiu : i ∈ u
    · rw [if_pos hiu, if_pos hiu]
      by_cases hig : i < g
      · rw [dif_pos hig, dif_pos hig]
        exact ih i hig hiu (fun g' hg' h' => hin g' hg' (h'.trans hig.le))
          fun g' hg' h' => hrd g' hg' (h'.trans hig.le)
      · rw [dif_neg hig, dif_neg hig]
    · rw [if_neg hiu, if_neg hiu]
      exact hrd g hg le_rfl x hx i hi hiu

theorem Prog.correct_of_congr (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ}
    (hA : ∀ g ∈ u, ref₁ g = ref₂ g)
    (hB : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → ref₁ i = ref₂ i)
    (h : C.Correct ref₁ u) : C.Correct ref₂ u := by
  intro g hg
  obtain ⟨h1, h2⟩ := h g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hb => ?_⟩
  · rw [← C.read_congr (hB g hg x hx i hi hiu)]
    exact h1 x hx i hi hiu
  · obtain ⟨b, hb1, hb2⟩ := h2 hb
    refine ⟨b, by rw [← hA g hg]; exact hb1, ?_⟩
    rw [← C.runIn_congr g hg (fun g' hg' _ _ => hA g' hg')
      fun g' hg' _ x hx i hi hiu => C.read_congr (hB g' hg' x hx i hi hiu)]
    exact hb2

/-- **A proof block's correctness reads only its gates and the gates they read.** -/
theorem Prog.correct_congr (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ}
    (hA : ∀ g ∈ u, ref₁ g = ref₂ g)
    (hB : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → ref₁ i = ref₂ i) :
    C.Correct ref₁ u ↔ C.Correct ref₂ u :=
  ⟨C.correct_of_congr hA hB,
    C.correct_of_congr (fun g hg => (hA g hg).symm) fun g hg x hx i hi hiu => (hB g hg x hx i hi hiu).symm⟩

/-! ## The boundary -/

theorem Prog.refIn_of_boundary {C : Prog} {r : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool} {g : ℕ}
    (h : C.Boundary r g = true) : C.refIn r val ι g = val g := by
  unfold Prog.refIn
  simp [h]

theorem Prog.refIn_of_not_mem {C : Prog} {r : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool} {g : ℕ}
    (h : g ∉ r) : C.refIn r val ι g = val g := by
  unfold Prog.refIn
  simp [h]

theorem Prog.boundary_of_input {C : Prog} (u : List ℕ) {g : ℕ} (h : C.gates[g]? = some .input) :
    C.Boundary u g = true := by
  unfold Prog.Boundary
  simp [h]

theorem Prog.boundary_of_out {C : Prog} (u : List ℕ) {g : ℕ} (h : g ∈ C.outs) : C.Boundary u g = true := by
  unfold Prog.Boundary
  simp [h]

/-- A gate read by a gate outside `u` is on `u`'s boundary. -/
theorem Prog.boundary_of_read {C : Prog} {u : List ℕ} {g g' : ℕ} {x : BGate} (hg' : g' ∉ u)
    (hlt : g' < C.gates.length) (hx : C.gates[g']? = some x) (hi : g ∈ x.reads) : C.Boundary u g = true := by
  have h : ((List.range C.gates.length).any fun g'' =>
      decide (g'' ∉ u) && ((C.gates[g'']?.map BGate.reads).getD []).contains g) = true :=
    List.any_eq_true.2 ⟨g', List.mem_range.2 hlt, by simp [hg', hx, hi]⟩
  unfold Prog.Boundary
  simp only [h, Bool.or_true]

/-! ## What a proof block's correctness needs -/

/-- **What `Correct` reads** of the committed values for the set of gates `u`: a gate of `u` whose value `u` commits
(`Boundary`), and a gate outside `u`, other than a constant, that one of `u`'s gates reads. -/
def Prog.Needs (C : Prog) (u : List ℕ) (g : ℕ) : Prop :=
  (g ∈ u ∧ C.Boundary u g = true) ∨
    ((∃ g' ∈ u, ∃ x : BGate, C.gates[g']? = some x ∧ g ∈ x.reads) ∧ g ∉ u ∧ ∀ b, C.gates[g]? ≠ some (.const b))

theorem Prog.read_congr_needs (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ}
    (h : ∀ g, C.Needs u g → ref₁ g = ref₂ g) {g : ℕ} (hg : g ∈ u) {x : BGate} (hx : C.gates[g]? = some x) {i : ℕ}
    (hi : i ∈ x.reads) (hiu : i ∉ u) : C.read ref₁ i = C.read ref₂ i := by
  by_cases hc : ∃ b, C.gates[i]? = some (.const b)
  · obtain ⟨b, hb⟩ := hc
    simp [Prog.read, hb]
  · exact C.read_congr (h i (.inr ⟨⟨g, hg, x, hx, hi⟩, hiu, fun b hb => hc ⟨b, hb⟩⟩))

/-- **A proof block's correctness reads only what it needs**: committed values that agree wherever `Needs` reads
leave `u` correct alike. -/
theorem Prog.correct_of_needs (C : Prog) {ref₁ ref₂ : ℕ → Option Bool} {u : List ℕ}
    (h : ∀ g, C.Needs u g → ref₁ g = ref₂ g) (hc : C.Correct ref₁ u) : C.Correct ref₂ u := by
  intro g hg
  obtain ⟨h1, h2⟩ := hc g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hb => ?_⟩
  · rw [← C.read_congr_needs h hg hx hi hiu]
    exact h1 x hx i hi hiu
  · obtain ⟨b, hb1, hb2⟩ := h2 hb
    refine ⟨b, by rw [← h g (.inl ⟨hg, hb⟩)]; exact hb1, ?_⟩
    rw [← C.runIn_congr g hg
      (fun g' hg' _ hin => h g' (.inl ⟨hg', C.boundary_of_input u hin⟩))
      fun g' hg' _ x hx i hi hiu => C.read_congr_needs h hg' hx hi hiu]
    exact hb2

end Claims.Sampling
