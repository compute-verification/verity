module

public import Definitions.Claims.Sampling

/-!
# What a proof block's correctness reads

`Prog.Correct ref u` reads `ref` only at `u`'s gates and the gates they read (`Prog.correct_congr`), and more finely
only where `Prog.Needs` says: at a gate of `u` whose value `u` commits, and at a gate outside `u`, other than a
constant, that one of `u`'s gates reads (`Prog.correct_of_needs`), which `Prog.needsList` lists (`Prog.needs_iff`).
It reads only opened values, so it survives filling in values `ref` leaves unopened (`Prog.correct_mono`).
`Prog.runIn` reads the inputs and the outside reads
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

/-! ## A correct proof block stays correct when unopened values are filled in -/

theorem mapM_rd_mono {rd rd' : ℕ → Option Bool} :
    ∀ {ws : List ℕ}, (∀ i ∈ ws, ∀ b, rd i = some b → rd' i = some b) → ∀ {bs : List Bool},
      ws.mapM rd = some bs → ws.mapM rd' = some bs
  | [], _, bs, h => h
  | i :: ws, hrd, bs, h => by
    rcases hri : rd i with _ | c
    · simp [List.mapM_cons, hri] at h
    · rcases hws : ws.mapM rd with _ | cs
      · simp [List.mapM_cons, hri, hws] at h
      · have h1 := hrd i (by simp) c hri
        have h2 := mapM_rd_mono (ws := ws) (fun j hj => hrd j (by simp [hj])) hws
        simp only [List.mapM_cons, hri, hws, Option.bind_eq_bind, Option.bind_some, Option.pure_def,
          Option.some.injEq] at h
        simp [List.mapM_cons, h1, h2, ← h]

theorem BGate.val_mono {rd rd' : ℕ → Option Bool} {own own' : Option Bool} (x : BGate)
    (hrd : ∀ i ∈ x.reads, ∀ b, rd i = some b → rd' i = some b) (hown : ∀ b, own = some b → own' = some b) {b : Bool}
    (h : x.val rd own = some b) : x.val rd' own' = some b := by
  cases x with
  | input => exact hown b h
  | const c => exact h
  | not i =>
    simp only [BGate.val] at h ⊢
    obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [hrd i (by simp [BGate.reads]) c hc]
    rfl
  | and ws =>
    simp only [BGate.val] at h ⊢
    obtain ⟨bs, hbs, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [mapM_rd_mono (fun i hi => hrd i (by simpa [BGate.reads] using hi)) hbs]
    rfl
  | xor ws =>
    simp only [BGate.val] at h ⊢
    obtain ⟨bs, hbs, rfl⟩ := Option.map_eq_some_iff.1 h
    rw [mapM_rd_mono (fun i hi => hrd i (by simpa [BGate.reads] using hi)) hbs]
    rfl

theorem Prog.read_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b) {i : ℕ}
    {b : Bool} (h : C.read ref i = some b) : C.read ref' i = some b := by
  by_cases hc : ∃ c, C.gates[i]? = some (.const c)
  · obtain ⟨c, hc⟩ := hc
    have hr : ∀ r : ℕ → Option Bool, C.read r i = some c := fun r => by simp [Prog.read, hc]
    rw [hr ref] at h
    rw [hr ref']
    exact h
  · have hc' : ∀ c, C.gates[i]? ≠ some (.const c) := fun c h' => hc ⟨c, h'⟩
    rw [C.read_of_not_const ref hc'] at h
    rw [C.read_of_not_const ref' hc']
    exact hext i b h

theorem Prog.runIn_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b)
    (u : List ℕ) : ∀ g b, C.runIn ref u g = some b → C.runIn ref' u g = some b := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro b h
  rw [Prog.runIn] at h ⊢
  rcases hx : C.gates[g]? with _ | x
  · rw [hx] at h
    cases h
  · rw [hx] at h
    simp only [Option.bind_some] at h ⊢
    refine BGate.val_mono x (fun i _ c hc => ?_) (hext g) h
    by_cases hiu : i ∈ u
    · simp only [hiu, ite_true] at hc ⊢
      by_cases hig : i < g
      · rw [dite_eq_left hig] at hc ⊢
        exact ih i hig c hc
      · rw [dite_eq_right hig] at hc
        cases hc
    · simp only [hiu, ite_false] at hc ⊢
      exact Prog.read_mono hext hc

theorem Prog.refIn_mono {C : Prog} {u : List ℕ} {ref ref' : ℕ → Option Bool} {ι : ℕ → Bool}
    (hext : ∀ g b, ref g = some b → ref' g = some b) :
    ∀ g b, C.refIn u ref ι g = some b → C.refIn u ref' ι g = some b := by
  intro g b h
  unfold Prog.refIn at h ⊢
  split_ifs at h ⊢
  · exact h
  · exact hext g b h

/-- **A correct proof block stays correct** at committed values that open every value `ref` opens, alike: it reads only
opened values. -/
theorem Prog.correct_mono {C : Prog} {ref ref' : ℕ → Option Bool} (hext : ∀ g b, ref g = some b → ref' g = some b)
    {u : List ℕ} (hc : C.Correct ref u) : C.Correct ref' u := by
  intro g hg
  obtain ⟨h1, h2⟩ := hc g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hbd => ?_⟩
  · obtain ⟨c, hc⟩ := Option.ne_none_iff_exists'.1 (h1 x hx i hi hiu)
    rw [Prog.read_mono hext hc]
    exact Option.some_ne_none _
  · obtain ⟨b, hb1, hb2⟩ := h2 hbd
    exact ⟨b, hext g b hb1, Prog.runIn_mono hext u g b hb2⟩

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

theorem Prog.mem_reads_getD (C : Prog) {g i : ℕ} :
    i ∈ (C.gates[g]?.map BGate.reads).getD [] ↔ ∃ x : BGate, C.gates[g]? = some x ∧ i ∈ x.reads := by
  cases C.gates[g]? with
  | none => simp
  | some x => simp

/-- **What a proof block's correctness reads, as a list**: its boundary gates, then each wire one of its gates reads
outside it that is no constant, so a reveal that opens these opens all of `Prog.Needs` (`needs_iff`). -/
def Prog.needsList (C : Prog) (u : List ℕ) : List ℕ :=
  u.filter (C.Boundary u) ++
    u.flatMap fun g => ((C.gates[g]?.map BGate.reads).getD []).filter fun i =>
      decide (i ∉ u) && decide (∀ b, C.gates[i]? ≠ some (.const b))

theorem Prog.needs_iff (C : Prog) (u : List ℕ) (g : ℕ) : C.Needs u g ↔ g ∈ C.needsList u := by
  unfold Prog.Needs Prog.needsList
  rw [List.mem_append, List.mem_filter, List.mem_flatMap]
  refine or_congr Iff.rfl ⟨fun ⟨⟨g', hg', hr⟩, hgu, hc⟩ => ⟨g', hg', ?_⟩, fun ⟨g', hg', hm⟩ => ?_⟩
  · rw [List.mem_filter, C.mem_reads_getD]
    simp only [Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨hr, hgu, hc⟩
  · rw [List.mem_filter, C.mem_reads_getD] at hm
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hm
    exact ⟨⟨g', hg', hm.1⟩, hm.2⟩

/-! ## A proof block is correct on an evaluation -/

/-- A gate on a topological program's boundary is one of its gates. -/
theorem Prog.lt_of_boundary {C : Prog} (htop : C.Topological) {u : List ℕ} {g : ℕ} (hB : C.Boundary u g = true) :
    g < C.gates.length := by
  unfold Prog.Boundary at hB
  simp only [Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true, List.mem_range,
    Bool.and_eq_true] at hB
  rcases hB with (hout | hin) | ⟨g', hg', -, hr⟩
  · exact htop.2 g hout
  · exact (List.getElem?_eq_some_iff.1 hin).1
  · obtain ⟨x, hx, hi⟩ := C.mem_reads_getD.1 hr
    exact lt_trans (htop.1 g' x hx g hi) hg'

/-- **A proof block runs to an evaluation**: where `v` evaluates each gate of `u` but its inputs on `v`'s values of the
wires it reads, `u`'s gates read only earlier gates of `u`, each value `u` reads from outside is `v`'s, and each input of
`u` is committed at `v`'s value, `u` computes `v` at each of its gates. -/
theorem Prog.runIn_of_eval (C : Prog) {ref : ℕ → Option Bool} {u : List ℕ} {v : ℕ → Bool}
    (htop : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∈ u → i < g)
    (hv : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → x ≠ .input → x.val (fun i => some (v i)) none = some (v g))
    (hout : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → C.read ref i = some (v i))
    (hin : ∀ g ∈ u, C.gates[g]? = some .input → ref g = some (v g)) :
    ∀ g ∈ u, g < C.gates.length → C.runIn ref u g = some (v g) := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg hlt
  rw [Prog.runIn]
  cases hx : C.gates[g]? with
  | none => exact absurd (List.getElem?_eq_none_iff.1 hx) (by omega)
  | some x =>
    by_cases hxi : x = .input
    · subst hxi
      exact hin g hg hx
    · rw [← hv g hg x hx hxi]
      refine BGate.val_congr (fun i hi => ?_) fun h => absurd h hxi
      by_cases hiu : i ∈ u
      · have hig := htop g hg x hx i hi hiu
        rw [ite_eq_left hiu, dite_eq_left hig]
        exact ih i hig hiu (lt_trans hig hlt)
      · rw [ite_eq_right hiu]
        exact hout g hg x hx i hi hiu

/-- **A proof block is correct on an evaluation**: on a topological program, where `v` evaluates each gate of `u` but
its inputs on `v`'s values of the wires it reads and holds each constant's own value, and the committed values `ref` are
`v`'s wherever `u`'s correctness reads them (`Needs`), `u` is correct on `ref`. -/
theorem Prog.correct_of_eval (C : Prog) {ref : ℕ → Option Bool} {u : List ℕ} {v : ℕ → Bool} (htop : C.Topological)
    (hv : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → x ≠ .input → x.val (fun i => some (v i)) none = some (v g))
    (hc : ∀ g b, C.gates[g]? = some (.const b) → v g = b)
    (href : ∀ g, C.Needs u g → ref g = some (v g)) : C.Correct ref u := by
  have hread : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → C.read ref i = some (v i) := by
    intro g hg x hx i hi hiu
    unfold Prog.read
    split
    · rename_i b hb
      rw [hc i b hb]
    · rename_i hk
      exact href i (Or.inr ⟨⟨g, hg, x, hx, hi⟩, hiu, fun b hb => hk b hb⟩)
  intro g hg
  refine ⟨fun x hx i hi hiu => ?_, fun hB => ⟨v g, href g (Or.inl ⟨hg, hB⟩), ?_⟩⟩
  · rw [hread g hg x hx i hi hiu]
    exact Option.some_ne_none _
  · exact C.runIn_of_eval (fun g _ x hx i hi _ => htop.1 g x hx i hi) hv hread
      (fun g hg hin => href g (Or.inl ⟨hg, Prog.boundary_of_input u hin⟩)) g hg (Prog.lt_of_boundary htop hB)

end Claims.Sampling
