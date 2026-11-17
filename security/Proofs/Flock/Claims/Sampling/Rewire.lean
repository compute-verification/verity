module

public import Proofs.Flock.Claims.Sampling.ProgReads

/-! A program with its reads sent through a map (`Prog.rewire`), and a partition cut to the gates the map leaves alone. -/

@[expose] public section

namespace Claims.Sampling

/-- **Gate `x`, every wire it reads sent through `ω`.** -/
def BGate.rewire (ω : ℕ → ℕ) : BGate → BGate
  | .input => .input
  | .const c => .const c
  | .not i => .not (ω i)
  | .and ws => .and (ws.map ω)
  | .xor ws => .xor (ws.map ω)

/-- **`C` rewired by `ω`, returning `outs`**: `C`'s gates, each reading wire `ω i` where it read `i`. -/
def Prog.rewire (C : Prog) (ω : ℕ → ℕ) (outs : List ℕ) : Prog := ⟨C.gates.map (BGate.rewire ω), outs⟩

/-- **The wired program**: `C` rewired by `ω`, returning those of `C`'s returned gates that `keep` holds. -/
def Prog.wired (C : Prog) (ω : ℕ → ℕ) (keep : ℕ → Bool) : Prog := C.rewire ω (C.outs.filter keep)

/-- **A set of gates cut to those `ω` leaves alone**: a gate `ω` moves is one the wired program reads at its image. -/
def cutBlock (ω : ℕ → ℕ) (u : List ℕ) : List ℕ := u.filter fun g => decide (ω g = g)

/-- **The cut partition**: `P`, each proof block cut to the gates `ω` leaves alone. -/
def Partition.cut (P : Partition) (ω : ℕ → ℕ) : Partition := ⟨P.classes.map (List.map (List.map (cutBlock ω)))⟩

theorem mem_cutBlock {ω : ℕ → ℕ} {u : List ℕ} {g : ℕ} : g ∈ cutBlock ω u ↔ g ∈ u ∧ ω g = g := by
  simp [cutBlock]

theorem BGate.reads_rewire (ω : ℕ → ℕ) (x : BGate) : (x.rewire ω).reads = x.reads.map ω := by
  cases x <;> simp [BGate.rewire, BGate.reads]

theorem BGate.rewire_eq_const {ω : ℕ → ℕ} {x : BGate} {c : Bool} : x.rewire ω = .const c ↔ x = .const c := by
  cases x <;> simp [BGate.rewire]

theorem BGate.rewire_eq_input {ω : ℕ → ℕ} {x : BGate} : x.rewire ω = .input ↔ x = .input := by
  cases x <;> simp [BGate.rewire]

theorem mapM_map_rd (ω : ℕ → ℕ) (rd : ℕ → Option Bool) :
    ∀ ws : List ℕ, (ws.map ω).mapM rd = ws.mapM fun i => rd (ω i)
  | [] => rfl
  | w :: ws => by simp only [List.map_cons, List.mapM_cons, mapM_map_rd ω rd ws]

theorem BGate.val_rewire (ω : ℕ → ℕ) (x : BGate) (rd : ℕ → Option Bool) (own : Option Bool) :
    (x.rewire ω).val rd own = x.val (fun i => rd (ω i)) own := by
  cases x <;> simp only [BGate.rewire, BGate.val, mapM_map_rd]

theorem Prog.rewire_gates (C : Prog) (ω : ℕ → ℕ) (outs : List ℕ) (g : ℕ) :
    (C.rewire ω outs).gates[g]? = C.gates[g]?.map (BGate.rewire ω) := by
  simp [Prog.rewire]

theorem Prog.gate_rewire {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} {g : ℕ} {y : BGate}
    (hy : (C.rewire ω outs).gates[g]? = some y) :
    ∃ x, C.gates[g]? = some x ∧ y = x.rewire ω ∧
      ∀ (v : ℕ → Bool) (own : Option Bool),
        y.val (fun i => some (v i)) own = x.val (fun i => some (v (ω i))) own := by
  rw [Prog.rewire_gates] at hy
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hy
  exact ⟨x, hx, rfl, fun v own => BGate.val_rewire ω x (fun i => some (v i)) own⟩

theorem Prog.rewire_const {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} {g : ℕ} {c : Bool} :
    (C.rewire ω outs).gates[g]? = some (.const c) ↔ C.gates[g]? = some (.const c) := by
  rw [Prog.rewire_gates]
  constructor
  · intro hc
    obtain ⟨y, hy, hyc⟩ := Option.map_eq_some_iff.1 hc
    rw [BGate.rewire_eq_const.1 hyc] at hy
    exact hy
  · intro hc
    simp [hc, BGate.rewire]

theorem Prog.read_rewire_of_fixed {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} {ref : ℕ → Option Bool} {i : ℕ}
    (hωi : ω i = i) : (C.rewire ω outs).read ref i = C.read (fun j => ref (ω j)) i := by
  by_cases hcst : ∃ c, C.gates[i]? = some (.const c)
  · obtain ⟨c, hcc⟩ := hcst
    have h1 : C.read (fun j => ref (ω j)) i = some c := by simp [Prog.read, hcc]
    have h2 : (C.rewire ω outs).read ref i = some c := by simp [Prog.read, Prog.rewire_const.2 hcc]
    rw [h1, h2]
  · have hnc : ∀ c, C.gates[i]? ≠ some (.const c) := fun c h' => hcst ⟨c, h'⟩
    have h1 : C.read (fun j => ref (ω j)) i = ref (ω i) := C.read_of_not_const _ hnc
    have h2 : (C.rewire ω outs).read ref i = ref i :=
      Prog.read_of_not_const _ ref fun c hc' => hnc c (Prog.rewire_const.1 hc')
    exact h2.trans ((congrArg ref hωi).symm.trans h1.symm)

theorem Prog.evaluates_rewire {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} {v₀ v : ℕ → Bool} {g : ℕ} {y : BGate}
    (hy : (C.rewire ω outs).gates[g]? = some y) (hyne : y ≠ .input)
    (hv₀ : ∀ x : BGate, C.gates[g]? = some x → x ≠ .input → x.val (fun i => some (v₀ i)) none = some (v₀ g))
    (hg : v g = v₀ g) (hrd : ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, v (ω i) = v₀ i) :
    y.val (fun i => some (v i)) none = some (v g) := by
  obtain ⟨x, hx, rfl, hval⟩ := Prog.gate_rewire hy
  have hxne : x ≠ .input := fun hxi => hyne (BGate.rewire_eq_input.2 hxi)
  have hcongr : x.val (fun i => some (v (ω i))) none = x.val (fun i => some (v₀ i)) none :=
    BGate.val_congr (fun i hi => congrArg some (hrd x hx i hi)) fun _ => rfl
  exact (hval v none).trans (hcongr.trans ((hv₀ x hx hxne).trans (congrArg some hg.symm)))

theorem Prog.needs_rewire_of_read {C : Prog} {ω : ℕ → ℕ} {outs u : List ℕ} {g' i : ℕ} {x : BGate} (hg' : g' ∈ u)
    (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) (hout : ω i ∉ u)
    (hnc : ∀ c, C.gates[ω i]? ≠ some (.const c)) : (C.rewire ω outs).Needs u (ω i) := by
  have hx' : (C.rewire ω outs).gates[g']? = some (x.rewire ω) := by
    rw [Prog.rewire_gates, hx, Option.map_some]
  have hi' : ω i ∈ (x.rewire ω).reads := by
    rw [BGate.reads_rewire]
    exact List.mem_map.2 ⟨i, hi, rfl⟩
  unfold Prog.Needs
  exact Or.inr ⟨⟨g', hg', x.rewire ω, hx', hi'⟩, hout, fun c hc => hnc c (Prog.rewire_const.1 hc)⟩

theorem Prog.boundary_rewire_of_read {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} (v : List ℕ) {g' i : ℕ} {x : BGate}
    (hg' : g' ∉ v) (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) :
    (C.rewire ω outs).Boundary v (ω i) = true := by
  have hx' : (C.rewire ω outs).gates[g']? = some (x.rewire ω) := by
    rw [Prog.rewire_gates, hx, Option.map_some]
  have hi' : ω i ∈ (x.rewire ω).reads := by
    rw [BGate.reads_rewire]
    exact List.mem_map.2 ⟨i, hi, rfl⟩
  exact Prog.boundary_of_read hg' (List.getElem?_eq_some_iff.1 hx').1 hx' hi'

theorem Prog.needs_rewire_of_written {C : Prog} {ω : ℕ → ℕ} {outs v : List ℕ} {g' i : ℕ} {x : BGate}
    (hw : ω i ∈ v) (hg' : g' ∉ v) (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) :
    (C.rewire ω outs).Needs v (ω i) := by
  unfold Prog.Needs
  exact Or.inl ⟨hw, Prog.boundary_rewire_of_read v hg' hx hi⟩

theorem Prog.needs_rewire_both {C : Prog} {ω : ℕ → ℕ} {outs u v : List ℕ} {g' i : ℕ} {x : BGate} (hg' : g' ∈ u)
    (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) (hw : ω i ∈ v) (hwu : ω i ∉ u) (hg'v : g' ∉ v)
    (hnc : ∀ c, C.gates[ω i]? ≠ some (.const c)) :
    (C.rewire ω outs).Needs u (ω i) ∧ (C.rewire ω outs).Needs v (ω i) :=
  ⟨Prog.needs_rewire_of_read hg' hx hi hwu hnc, Prog.needs_rewire_of_written hw hg'v hx hi⟩

theorem Prog.boundary_eq_false {C : Prog} {u : List ℕ} {g : ℕ} (ho : g ∉ C.outs)
    (hi : C.gates[g]? ≠ some .input) (hr : ∀ g' (x : BGate), C.gates[g']? = some x → g ∈ x.reads → g' ∈ u) :
    C.Boundary u g = false := by
  cases hB : C.Boundary u g
  · rfl
  exfalso
  simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
    List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not] at hB
  rcases hB with (hout | hin) | ⟨g', -, hg'u, hrd⟩
  · exact ho hout
  · exact hi hin
  · obtain ⟨x, hx, hix⟩ := C.mem_reads_getD.1 hrd
    exact hg'u (hr g' x hx hix)

theorem Prog.boundary_rewire_eq_false {C : Prog} {ω : ℕ → ℕ} {outs rb : List ℕ} {w : ℕ} (ho : w ∉ outs)
    (hi : C.gates[w]? ≠ some .input)
    (hr : ∀ g' (x : BGate) (i : ℕ), C.gates[g']? = some x → i ∈ x.reads → ω i = w → g' ∈ rb) :
    (C.rewire ω outs).Boundary rb w = false := by
  have ho' : w ∉ (C.rewire ω outs).outs := ho
  refine Prog.boundary_eq_false ho' (fun hin => ?_) fun g' y hy hw => ?_
  · rw [Prog.rewire_gates] at hin
    obtain ⟨x, hx, hxi⟩ := Option.map_eq_some_iff.1 hin
    rw [BGate.rewire_eq_input.1 hxi] at hx
    exact hi hx
  · obtain ⟨x, hx, rfl, -⟩ := Prog.gate_rewire hy
    rw [BGate.reads_rewire] at hw
    obtain ⟨i, hix, hiw⟩ := List.mem_map.1 hw
    exact hr g' x i hx hix hiw

theorem Prog.boundary_wired_eq_false {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {rb : List ℕ} {w : ℕ}
    (hk : keep w = false) (hi : C.gates[w]? ≠ some .input)
    (hr : ∀ g' (x : BGate) (i : ℕ), C.gates[g']? = some x → i ∈ x.reads → ω i = w → g' ∈ rb) :
    (C.wired ω keep).Boundary rb w = false := by
  unfold Prog.wired
  refine Prog.boundary_rewire_eq_false (fun h => ?_) hi hr
  have h2 := (List.mem_filter.1 h).2
  rw [hk] at h2
  cases h2

theorem Prog.unread_rewire {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} {s : ℕ} (hin : C.gates[s]? = some .input)
    (ho : s ∉ outs) (hω : ∀ x ∈ C.gates, ∀ i ∈ x.reads, ω i ≠ s) : (C.rewire ω outs).Unread s := by
  unfold Prog.Unread
  have ho' : s ∉ (C.rewire ω outs).outs := ho
  refine ⟨?_, ho', fun y hy hs => ?_⟩
  · simp [Prog.rewire_gates, hin, BGate.rewire]
  · have hy' : y ∈ C.gates.map (BGate.rewire ω) := hy
    obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hy'
    rw [BGate.reads_rewire] at hs
    obtain ⟨i, hi, his⟩ := List.mem_map.1 hs
    exact hω x hx i hi his

theorem Prog.not_needs_of_unread {C : Prog} {u : List ℕ} {g : ℕ} (hg : g ∉ u) (hr : ∀ x ∈ C.gates, g ∉ x.reads) :
    ¬ C.Needs u g := by
  unfold Prog.Needs
  rintro (⟨hgu, -⟩ | ⟨⟨g', -, x, hx, hi⟩, -⟩)
  · exact hg hgu
  · exact hr x (List.mem_of_getElem? hx) hi

theorem Prog.runIn_rewire {C : Prog} {ω : ℕ → ℕ} {outs u u' : List ℕ} {ref₀ ref : ℕ → Option Bool}
    (hsub : ∀ g ∈ u', g ∈ u) (hid : ∀ g ∈ u', ω g = g) (href : ∀ g ∈ u', ref₀ g = ref g)
    (hrd : ∀ g ∈ u', ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u' →
      ω i ∉ u' ∧ (i ∈ u → C.gates[i]? = some .input) ∧
        ∀ b, C.read ref₀ i = some b → (C.rewire ω outs).read ref (ω i) = some b) :
    ∀ g ∈ u', ∀ b, C.runIn ref₀ u g = some b → (C.rewire ω outs).runIn ref u' g = some b := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
  intro hg b h
  rw [Prog.runIn] at h ⊢
  rcases hx : C.gates[g]? with _ | x
  · rw [hx] at h
    cases h
  · rw [hx] at h
    have hx' : (C.rewire ω outs).gates[g]? = some (x.rewire ω) := by
      rw [Prog.rewire_gates, hx, Option.map_some]
    rw [hx']
    simp only [Option.bind_some] at h ⊢
    rw [BGate.val_rewire]
    refine BGate.val_mono x (fun i hi c hc => ?_) (fun c hc => (href g hg).symm.trans hc) h
    by_cases hiu' : i ∈ u'
    · simp only [hsub i hiu', ite_true] at hc
      simp only [hid i hiu', hiu', ite_true]
      by_cases hig : i < g
      · rw [dite_eq_left hig] at hc ⊢
        exact ih i hig hiu' c hc
      · rw [dite_eq_right hig] at hc
        cases hc
    · obtain ⟨hω, hin, hval⟩ := hrd g hg x hx i hi hiu'
      simp only [hω, ite_false]
      by_cases hiu : i ∈ u
      · simp only [hiu, ite_true] at hc
        have hnc : ∀ b, C.gates[i]? ≠ some (.const b) := fun b => by rw [hin hiu]; simp
        by_cases hig : i < g
        · rw [dite_eq_left hig, Prog.runIn, hin hiu] at hc
          simp [BGate.val] at hc
          exact hval c ((C.read_of_not_const ref₀ hnc).trans hc)
        · rw [dite_eq_right hig] at hc
          cases hc
      · simp only [hiu, ite_false] at hc
        exact hval c hc

theorem Prog.correct_rewire {C : Prog} {ω : ℕ → ℕ} {outs u u' : List ℕ} {ref₀ ref : ℕ → Option Bool}
    (hsub : ∀ g ∈ u', g ∈ u) (hid : ∀ g ∈ u', ω g = g) (href : ∀ g ∈ u', ref₀ g = ref g)
    (hrd : ∀ g ∈ u', ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u' →
      ω i ∉ u' ∧ (i ∈ u → C.gates[i]? = some .input) ∧
        ∀ b, C.read ref₀ i = some b → (C.rewire ω outs).read ref (ω i) = some b)
    (hbd : ∀ g ∈ u', (C.rewire ω outs).Boundary u' g = true → C.Boundary u g = true)
    (hc : C.Correct ref₀ u) : (C.rewire ω outs).Correct ref u' := by
  intro g hg
  obtain ⟨h1, h2⟩ := hc g (hsub g hg)
  refine ⟨fun y hy j hj hju => ?_, fun hb => ?_⟩
  · obtain ⟨x, hx, rfl, -⟩ := Prog.gate_rewire hy
    rw [BGate.reads_rewire] at hj
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hj
    have hiu' : i ∉ u' := fun h => hju (by rw [hid i h]; exact h)
    obtain ⟨-, hin, hval⟩ := hrd g hg x hx i hi hiu'
    have hopen : C.read ref₀ i ≠ none := by
      by_cases hiu : i ∈ u
      · have hnc : ∀ b, C.gates[i]? ≠ some (.const b) := fun b => by rw [hin hiu]; simp
        obtain ⟨b, hb1, -⟩ := (hc i hiu).2 (Prog.boundary_of_input u (hin hiu))
        rw [C.read_of_not_const ref₀ hnc, hb1]
        exact Option.some_ne_none _
      · exact h1 x hx i hi hiu
    obtain ⟨c, hc'⟩ := Option.ne_none_iff_exists'.1 hopen
    rw [hval c hc']
    exact Option.some_ne_none _
  · obtain ⟨b, hb1, hb2⟩ := h2 (hbd g hg hb)
    exact ⟨b, (href g hg).symm.trans hb1, Prog.runIn_rewire hsub hid href hrd g hg b hb2⟩

theorem Prog.boundary_of_boundary_rewire {C : Prog} {ω : ℕ → ℕ} {outs u u' : List ℕ} {g : ℕ}
    (ho : ∀ o ∈ outs, o ∈ C.outs) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (hsrc : ∀ g' ∈ u, g' ∉ u' → C.gates[g']? = some .input)
    (hb : (C.rewire ω outs).Boundary u' g = true) : C.Boundary u g = true := by
  simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
    List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not] at hb
  rcases hb with (hout | hin) | ⟨g', -, hg'u', hrd⟩
  · exact Prog.boundary_of_out u (ho g hout)
  · rw [Prog.rewire_gates] at hin
    obtain ⟨x, hx, hxi⟩ := Option.map_eq_some_iff.1 hin
    rw [BGate.rewire_eq_input.1 hxi] at hx
    exact Prog.boundary_of_input u hx
  · obtain ⟨y, hy, hgy⟩ := (C.rewire ω outs).mem_reads_getD.1 hrd
    obtain ⟨x, hx, rfl, -⟩ := Prog.gate_rewire hy
    rw [BGate.reads_rewire] at hgy
    obtain ⟨i, hi, hig⟩ := List.mem_map.1 hgy
    by_cases hωi : ω i = i
    · have hgx : g ∈ x.reads := by
        rw [← hig, hωi]
        exact hi
      have hg'u : g' ∉ u := fun h => by
        have hinp := hsrc g' h hg'u'
        rw [hx] at hinp
        obtain rfl := Option.some.inj hinp
        simp [BGate.reads] at hgx
      exact Prog.boundary_of_read hg'u (List.getElem?_eq_some_iff.1 hx).1 hx hgx
    · rw [← hig]
      exact Prog.boundary_of_out u (hw i hωi)

theorem Prog.needs_rewire {C : Prog} {ω : ℕ → ℕ} {outs u : List ℕ} {g : ℕ}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (ho : ∀ o ∈ outs, o ∈ C.outs) (hn : (C.rewire ω outs).Needs (cutBlock ω u) g) :
    ∃ i, C.Needs u i ∧ ω i = g := by
  unfold Prog.Needs at hn
  rcases hn with ⟨hgu, hb⟩ | ⟨⟨g', hg', y, hy, hgy⟩, hgu, hnc⟩
  · have hg := mem_cutBlock.1 hgu
    refine ⟨g, ?_, hg.2⟩
    unfold Prog.Needs
    exact Or.inl ⟨hg.1, Prog.boundary_of_boundary_rewire ho hw
      (fun g' hg'u hg'u' => hin g' fun h => hg'u' (mem_cutBlock.2 ⟨hg'u, h⟩)) hb⟩
  · obtain ⟨x, hx, rfl, -⟩ := Prog.gate_rewire hy
    rw [BGate.reads_rewire] at hgy
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hgy
    have hg'u : g' ∈ u := (mem_cutBlock.1 hg').1
    refine ⟨i, ?_, rfl⟩
    unfold Prog.Needs
    by_cases hiu : i ∈ u
    · have hωi : ω i ≠ i := fun h => hgu (by rw [h]; exact mem_cutBlock.2 ⟨hiu, h⟩)
      exact Or.inl ⟨hiu, Prog.boundary_of_input u (hin i hωi)⟩
    · refine Or.inr ⟨⟨g', hg'u, x, hx, hi⟩, hiu, fun c hc => ?_⟩
      have hωi : ω i = i := by
        by_contra hne
        have hinp := hin i hne
        rw [hc] at hinp
        cases hinp
      exact hnc c (by rw [hωi]; exact Prog.rewire_const.2 hc)

theorem Prog.topological_rewire {C : Prog} {ω : ℕ → ℕ} {outs : List ℕ} (htop : C.Topological)
    (hω : ∀ i, ω i ≤ i) (ho : ∀ o ∈ outs, o ∈ C.outs) : (C.rewire ω outs).Topological := by
  unfold Prog.Topological at htop ⊢
  refine ⟨fun g y hy j hj => ?_, fun o ho' => ?_⟩
  · obtain ⟨x, hx, rfl, -⟩ := Prog.gate_rewire hy
    rw [BGate.reads_rewire] at hj
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hj
    exact lt_of_le_of_lt (hω i) (htop.1 g x hx i hi)
  · have hlen : (C.rewire ω outs).gates.length = C.gates.length := by simp [Prog.rewire]
    rw [hlen]
    exact htop.2 o (ho o ho')

theorem Partition.blocks_cut (P : Partition) (ω : ℕ → ℕ) : (P.cut ω).blocks = P.blocks.map (cutBlock ω) := by
  unfold Partition.cut Partition.blocks
  rw [List.map_flatten, List.map_flatten]

theorem Partition.cut_getElem? (P : Partition) (ω : ℕ → ℕ) (v : ℕ) :
    (P.cut ω).blocks[v]? = P.blocks[v]?.map (cutBlock ω) := by
  rw [Partition.blocks_cut, List.getElem?_map]

theorem Partition.length_blocks_cut (P : Partition) (ω : ℕ → ℕ) : (P.cut ω).blocks.length = P.blocks.length := by
  rw [Partition.blocks_cut, List.length_map]

theorem flatten_map_cutBlock (ω : ℕ → ℕ) : ∀ L : List (List ℕ), (L.map (cutBlock ω)).flatten = cutBlock ω L.flatten
  | [] => rfl
  | u :: L => by
    rw [List.map_cons, List.flatten_cons, List.flatten_cons, flatten_map_cutBlock ω L]
    unfold cutBlock
    rw [List.filter_append]

theorem Partition.gates_cut (P : Partition) (ω : ℕ → ℕ) : (P.cut ω).gates = cutBlock ω P.gates := by
  unfold Partition.gates
  rw [Partition.blocks_cut, flatten_map_cutBlock]

theorem Partition.nodup_gates_cut {P : Partition} (ω : ℕ → ℕ) (h : P.gates.Nodup) : (P.cut ω).gates.Nodup := by
  rw [Partition.gates_cut]
  unfold cutBlock
  exact h.filter _

theorem getD_blocks_cut (P : Partition) (ω : ℕ → ℕ) (w : ℕ) :
    (P.cut ω).blocks.getD w [] = cutBlock ω (P.blocks.getD w []) := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, Partition.cut_getElem?]
  cases P.blocks[w]? with
  | none => rfl
  | some u => rfl

theorem cutBlock_flatMap (ω : ℕ → ℕ) (f : ℕ → List ℕ) :
    ∀ l : List ℕ, cutBlock ω (l.flatMap f) = l.flatMap fun w => cutBlock ω (f w)
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons, List.flatMap_cons, ← cutBlock_flatMap ω f l]
    unfold cutBlock
    rw [List.filter_append]

theorem Partition.cut_union {Pt Pρ : Partition} (ω : ℕ → ℕ) {I : ℕ → List ℕ} {ρ : ℕ}
    (hρ : Pρ.blocks[ρ]? = some ((I ρ).flatMap fun w => Pt.blocks.getD w [])) :
    (Pρ.cut ω).blocks[ρ]? = some ((I ρ).flatMap fun w => (Pt.cut ω).blocks.getD w []) := by
  rw [Partition.cut_getElem?, hρ, Option.map_some, cutBlock_flatMap]
  simp only [getD_blocks_cut]

theorem strataSizes_map_cut (ω : ℕ → ℕ) (c : List (List (List ℕ))) :
    strataSizes (c.map (List.map (cutBlock ω))) = strataSizes c := by
  unfold strataSizes
  rw [List.map_map]
  exact List.map_congr_left fun s _ => by simp

theorem Partition.shape_cut (P : Partition) (ω : ℕ → ℕ) :
    (P.cut ω).classes.map strataSizes = P.classes.map strataSizes := by
  show (P.classes.map (List.map (List.map (cutBlock ω)))).map strataSizes = P.classes.map strataSizes
  rw [List.map_map]
  exact List.map_congr_left fun c _ => strataSizes_map_cut ω c

theorem Prog.correct_wired {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {u : List ℕ} {ref : ℕ → Option Bool}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hnc : ∀ i, ω i ≠ i → ∀ c, C.gates[ω i]? ≠ some (.const c))
    (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (htgt : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, ω i ≠ i → ω i ∉ u)
    (hc : C.Correct (fun i => ref (ω i)) u) : (C.wired ω keep).Correct ref (cutBlock ω u) := by
  unfold Prog.wired
  have hsub : ∀ g ∈ cutBlock ω u, g ∈ u := fun g hg => (mem_cutBlock.1 hg).1
  have hid : ∀ g ∈ cutBlock ω u, ω g = g := fun g hg => (mem_cutBlock.1 hg).2
  refine Prog.correct_rewire (ref₀ := fun i => ref (ω i)) hsub hid (fun g hg => congrArg ref (hid g hg))
    (fun g hg x hx i hi hiu' => ?_) (fun g hg hb => ?_) hc
  · by_cases hωi : ω i = i
    · have hiu : i ∉ u := fun h => hiu' (mem_cutBlock.2 ⟨h, hωi⟩)
      refine ⟨?_, fun h => absurd h hiu, fun b hb => ?_⟩
      · rw [hωi]
        exact hiu'
      · rw [hωi]
        exact (Prog.read_rewire_of_fixed hωi).trans hb
    · have hinp := hin i hωi
      have h1 : C.read (fun j => ref (ω j)) i = ref (ω i) :=
        C.read_of_not_const _ fun c => by rw [hinp]; simp
      have h2 : (C.rewire ω (C.outs.filter keep)).read ref (ω i) = ref (ω i) :=
        Prog.read_of_not_const _ ref fun c hc' => hnc i hωi c (Prog.rewire_const.1 hc')
      exact ⟨fun h => htgt g (hsub g hg) x hx i hi hωi (hsub _ h), fun _ => hinp,
        fun b hb => h2.trans (h1.symm.trans hb)⟩
  · exact Prog.boundary_of_boundary_rewire (fun o ho => (List.mem_filter.1 ho).1) hw
      (fun g' hg'u hg'u' => hin g' fun h => hg'u' (mem_cutBlock.2 ⟨hg'u, h⟩)) hb

theorem Prog.correct_wired_of_agree {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {u : List ℕ} {v₀ v : ℕ → Bool}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hnc : ∀ i, ω i ≠ i → ∀ c, C.gates[ω i]? ≠ some (.const c))
    (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (htgt : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, ω i ≠ i → ω i ∉ u)
    (hu : ∀ g ∈ u, v (ω g) = v₀ g)
    (hr : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, i ∉ u → v (ω i) = v₀ i)
    (hc : C.Correct (fun g => some (v₀ g)) u) :
    (C.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω u) := by
  refine Prog.correct_wired hin hnc hw htgt
    (C.correct_of_congr (fun g hg => ?_) (fun g hg x hx i hi hiu => ?_) hc)
  · show some (v₀ g) = some (v (ω g))
    rw [hu g hg]
  · show some (v₀ i) = some (v (ω i))
    rw [hr g hg x hx i hi hiu]

theorem Prog.exists_correct_wired {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {u : List ℕ} {v₀ : ℕ → Bool}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hnc : ∀ i, ω i ≠ i → ∀ c, C.gates[ω i]? ≠ some (.const c))
    (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (htgt : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, ω i ≠ i → ω i ∉ u)
    (S : ℕ → Prop) (hSu : ∀ g ∈ u, S g)
    (hSr : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, S i)
    (hagree : ∀ i j, S i → S j → ω i = ω j → v₀ i = v₀ j)
    (hc : C.Correct (fun g => some (v₀ g)) u) :
    ∃ v : ℕ → Bool, (∀ i, S i → v (ω i) = v₀ i) ∧
      (C.wired ω keep).Correct (fun g => some (v g)) (cutBlock ω u) := by
  have hex : ∃ v : ℕ → Bool, ∀ i, S i → v (ω i) = v₀ i := by
    classical
    refine ⟨fun g => if h : ∃ j, S j ∧ ω j = g then v₀ h.choose else false, fun i hi => ?_⟩
    have h : ∃ j, S j ∧ ω j = ω i := ⟨i, hi, rfl⟩
    show (if h : ∃ j, S j ∧ ω j = ω i then v₀ h.choose else false) = v₀ i
    rw [dite_eq_left h]
    exact hagree _ _ h.choose_spec.1 hi h.choose_spec.2
  obtain ⟨v, hv⟩ := hex
  exact ⟨v, hv, Prog.correct_wired_of_agree hin hnc hw htgt (fun g hg => hv g (hSu g hg))
    (fun g hg x hx i hi _ => hv i (hSr g hg x hx i hi)) hc⟩

theorem Prog.needs_wired {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {u : List ℕ} {g : ℕ}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (hn : (C.wired ω keep).Needs (cutBlock ω u) g) : ∃ i, C.Needs u i ∧ ω i = g := by
  unfold Prog.wired at hn
  exact Prog.needs_rewire hin hw (fun o ho => (List.mem_filter.1 ho).1) hn

theorem Prog.fixed_of_needs_wired {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} {u : List ℕ} {g : ℕ}
    (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input) (hw : ∀ i, ω i ≠ i → ω i ∈ C.outs)
    (hidem : ∀ i, ω (ω i) = ω i) (hn : (C.wired ω keep).Needs (cutBlock ω u) g) : ω g = g := by
  obtain ⟨i, -, rfl⟩ := Prog.needs_wired hin hw hn
  exact hidem i

theorem Prog.topological_wired {C : Prog} {ω : ℕ → ℕ} {keep : ℕ → Bool} (htop : C.Topological)
    (hω : ∀ i, ω i ≤ i) : (C.wired ω keep).Topological := by
  unfold Prog.wired
  exact Prog.topological_rewire htop hω fun o ho => (List.mem_filter.1 ho).1

theorem le_of_earlier {ω mem : ℕ → ℕ} (hmem : ∀ i j, mem i < mem j → i < j)
    (hω : ∀ i, ω i ≠ i → mem (ω i) < mem i) (i : ℕ) : ω i ≤ i := by
  by_cases h : ω i = i
  · exact Nat.le_of_eq h
  · exact Nat.le_of_lt (hmem _ _ (hω i h))

theorem idem_of_not_input {C : Prog} {ω : ℕ → ℕ} (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hni : ∀ i, ω i ≠ i → C.gates[ω i]? ≠ some .input) (i : ℕ) : ω (ω i) = ω i := by
  by_cases h : ω i = i
  · exact congrArg ω h
  · exact Classical.byContradiction fun h2 => hni i h (hin (ω i) h2)

theorem off_of_earlier {C : Prog} {ω mem : ℕ → ℕ} {u : List ℕ} {t : ℕ} (hu : ∀ g ∈ u, mem g = t)
    (hr : ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, mem i = t)
    (hω : ∀ i, ω i ≠ i → mem (ω i) < mem i) :
    ∀ g ∈ u, ∀ x : BGate, C.gates[g]? = some x → ∀ i ∈ x.reads, ω i ≠ i → ω i ∉ u := by
  intro g hg x hx i hi hne hmem
  have h1 : mem (ω i) < mem i := hω i hne
  rw [hu _ hmem, hr g hg x hx i hi] at h1
  exact Nat.lt_irrefl t h1

theorem stays_of_not_input {C : Prog} {ω : ℕ → ℕ} {u : List ℕ} (hin : ∀ i, ω i ≠ i → C.gates[i]? = some .input)
    (hu : ∃ g ∈ u, C.gates[g]? ≠ some .input) : ∃ g ∈ u, ω g = g := by
  obtain ⟨g, hg, hne⟩ := hu
  exact ⟨g, hg, Classical.byContradiction fun h => hne (hin g h)⟩

end Claims.Sampling
