import Definitions.Claims.Sampling

/-!
# The sampling block's faulty set is the boundary form's

`Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary`. On a topologically ordered program, a replay unit
whose proof units are disjoint sets of gates has an interior making every proof unit correct exactly when the set of its
gates is correct on the committed values. Both ways, a proof unit computes what the replay unit's gates compute, once
every value it reads from another of its replay unit's proof units is the one the replay unit computes (`runIn_step`).
One way that value is on the other unit's boundary, so the other unit's check commits it (`runIn_units`); the other way,
the replay unit's own computed values are the interior (`ι g = runIn val S g`, `units_of_correct`).
-/

namespace Claims.Sampling.Boundary

open Claims.Sampling

variable (C : Prog)

theorem runIn_eq (ref : ℕ → Option Bool) (u : List ℕ) (g : ℕ) :
    C.runIn ref u g = C.gates[g]?.bind fun x =>
      x.val (fun i => if i ∈ u then (if i < g then C.runIn ref u i else none) else C.read ref i) (ref g) := by
  rw [Prog.runIn]
  simp only [dite_eq_ite]

theorem mapM_congr' {f f' : ℕ → Option Bool} : ∀ (ws : List ℕ), (∀ i ∈ ws, f i = f' i) → ws.mapM f = ws.mapM f'
  | [], _ => rfl
  | w :: ws, h => by
    simp only [List.mapM_cons, h w List.mem_cons_self, mapM_congr' ws fun i hi => h i (List.mem_cons_of_mem _ hi)]

theorem mapM_isSome {f : ℕ → Option Bool} : ∀ (ws : List ℕ), (∀ i ∈ ws, (f i).isSome) → (ws.mapM f).isSome
  | [], _ => rfl
  | w :: ws, h => by
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.1 (h w List.mem_cons_self)
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.1 (mapM_isSome ws fun i hi => h i (List.mem_cons_of_mem _ hi))
    simp [List.mapM_cons, hb, hbs]

theorem val_congr (x : BGate) {rd rd' : ℕ → Option Bool} {own own' : Option Bool} (h : ∀ i ∈ x.reads, rd i = rd' i)
    (ho : x = .input → own = own') : x.val rd own = x.val rd' own' := by
  cases x with
  | input => exact ho rfl
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr' ws h]
  | xor ws => simp only [BGate.val, mapM_congr' ws h]

theorem val_isSome (x : BGate) (hx : x ≠ .input) {rd : ℕ → Option Bool} (own : Option Bool)
    (h : ∀ i ∈ x.reads, (rd i).isSome) : (x.val rd own).isSome := by
  cases x with
  | input => exact absurd rfl hx
  | const b => rfl
  | not i => simpa [BGate.val] using h i (by simp [BGate.reads])
  | and ws => simpa [BGate.val] using mapM_isSome ws h
  | xor ws => simpa [BGate.val] using mapM_isSome ws h

theorem read_const {ref : ℕ → Option Bool} {i : ℕ} {b : Bool} (h : C.gates[i]? = some (.const b)) :
    C.read ref i = some b := by
  simp [Prog.read, h]

theorem read_eq {ref : ℕ → Option Bool} {i : ℕ} (h : ∀ b, C.gates[i]? ≠ some (.const b)) : C.read ref i = ref i := by
  cases hg : C.gates[i]? with
  | none => simp [Prog.read, hg]
  | some x => cases x <;> first | exact absurd hg (h _) | simp [Prog.read, hg]

theorem read_congr {ρ ρ' : ℕ → Option Bool} {i : ℕ} (h : ρ i = ρ' i) : C.read ρ i = C.read ρ' i := by
  by_cases hc : ∃ b, C.gates[i]? = some (.const b)
  · obtain ⟨b, hb⟩ := hc
    rw [read_const C hb, read_const C hb]
  · push Not at hc
    rw [read_eq C hc, read_eq C hc, h]

/-- A constant computes its own value in any set of gates. -/
theorem runIn_const {ref : ℕ → Option Bool} {u : List ℕ} {i : ℕ} {b : Bool} (h : C.gates[i]? = some (.const b)) :
    C.runIn ref u i = some b := by
  rw [runIn_eq, h]
  rfl

theorem boundary_input {u : List ℕ} {g : ℕ} (h : C.gates[g]? = some .input) : C.Boundary u g = true := by
  simp [Prog.Boundary, h]

theorem boundary_mono {u S : List ℕ} {g : ℕ} (hsub : ∀ i ∈ u, i ∈ S) (h : C.Boundary S g = true) :
    C.Boundary u g = true := by
  simp only [Prog.Boundary, Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h ⊢
  rcases h with (h | h) | ⟨g', hg', h1, h2⟩
  · exact Or.inl (Or.inl h)
  · exact Or.inl (Or.inr h)
  · exact Or.inr ⟨g', hg', fun h' => h1 (hsub g' h'), h2⟩

theorem boundary_of_read {u : List ℕ} {g i : ℕ} {x : BGate} (hg : C.gates[g]? = some x) (hi : i ∈ x.reads)
    (hout : g ∉ u) : C.Boundary u i = true := by
  have hlen : g < C.gates.length := by
    rcases Nat.lt_or_ge g C.gates.length with h | h
    · exact h
    · simp [List.getElem?_eq_none h] at hg
  simp only [Prog.Boundary, Bool.or_eq_true, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq]
  exact Or.inr ⟨g, List.mem_range.2 hlen, hout, by simp [hg, hi]⟩

theorem ref_out {r : RU} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : i ∉ RU.gates r) :
    RU.ref C r val ι i = val i := by
  rw [RU.ref, ite_eq_right fun h' => h h'.1]

theorem ref_boundary {r : RU} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : C.Boundary (RU.gates r) i = true) :
    RU.ref C r val ι i = val i := by
  rw [RU.ref, ite_eq_right fun h' => by simp [h] at h']

theorem ref_input {r : RU} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : C.gates[i]? = some .input) :
    RU.ref C r val ι i = val i :=
  ref_boundary C (boundary_input C h)

theorem sub_gates {r : RU} {u : List ℕ} (hu : u ∈ r) : ∀ i ∈ u, i ∈ RU.gates r :=
  fun _ hi => List.mem_flatten.2 ⟨u, hu, hi⟩

/-- Disjoint proof units: a gate of `u` is in no other proof unit `u'` of the replay unit. -/
theorem not_mem_other {r : RU} (hnd : (RU.gates r).Nodup) {u u' : List ℕ} (hu : u ∈ r) (hu' : u' ∈ r) {i g : ℕ}
    (hiu' : i ∈ u') (hiu : i ∉ u) (hgu : g ∈ u) : g ∉ u' := by
  have hne : u ≠ u' := fun h => hiu (h ▸ hiu')
  have : Std.Symm (List.Disjoint (α := ℕ)) := ⟨fun _ _ h => List.Disjoint.symm h⟩
  exact fun hgu' => (List.nodup_flatten.1 hnd).2.forall hu hu' hne hgu hgu'

/-- **A gate computes in the proof unit `u` what it computes in `S ⊇ u`** when every wire it reads agrees: one of `u` as
computed in each, one of `S` outside `u` read in `u` as the value `S` computes, and one outside `S` read the same. -/
theorem runIn_step (htop : C.Topological) {ρ val : ℕ → Option Bool} {u S : List ℕ} {g : ℕ}
    (hsub : ∀ i ∈ u, i ∈ S) (hin : C.gates[g]? = some .input → ρ g = val g)
    (hrd : ∀ x, C.gates[g]? = some x → ∀ i ∈ x.reads,
      (i ∈ u → C.runIn ρ u i = C.runIn val S i) ∧ (i ∉ u → i ∈ S → C.read ρ i = C.runIn val S i) ∧
        (i ∉ S → C.read ρ i = C.read val i)) :
    C.runIn ρ u g = C.runIn val S g := by
  rw [runIn_eq, runIn_eq]
  cases hx : C.gates[g]? with
  | none => rfl
  | some x =>
    simp only [Option.bind_some]
    refine val_congr x (fun i hi => ?_) (fun h => hin (by rw [hx, h]))
    have hig : i < g := htop.1 g x hx i hi
    obtain ⟨h1, h2, h3⟩ := hrd x hx i hi
    by_cases hiu : i ∈ u
    · simp [hiu, hsub i hiu, hig, h1 hiu]
    · by_cases hiS : i ∈ S
      · simp [hiu, hiS, hig, h2 hiu hiS]
      · simp [hiu, hiS, h3 hiS]

/-- **Correct proof units compute what their replay unit computes**: a value one reads from another is on that one's
boundary, so its check commits the value it computes. -/
theorem runIn_units (htop : C.Topological) {r : RU} (hnd : (RU.gates r).Nodup) {val : ℕ → Option Bool}
    {ι : ℕ → Bool} (hcor : ∀ u ∈ r, C.Correct (RU.ref C r val ι) u) :
    ∀ g, ∀ u ∈ r, g ∈ u → C.runIn (RU.ref C r val ι) u g = C.runIn val (RU.gates r) g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    intro u hu hgu
    refine runIn_step C htop (sub_gates hu) (ref_input C) fun x hx i hi =>
      ⟨fun hiu => ih i (htop.1 g x hx i hi) u hu hiu, fun hiu hiS => ?_, fun hiS => read_congr C (ref_out C hiS)⟩
    by_cases hc : ∃ b, C.gates[i]? = some (.const b)
    · obtain ⟨b, hb⟩ := hc
      rw [read_const C hb, runIn_const C hb]
    · push Not at hc
      obtain ⟨u', hu', hiu'⟩ := List.mem_flatten.1 hiS
      obtain ⟨b, hρ, hrun⟩ :=
        (hcor u' hu' i hiu').2 (boundary_of_read C hx hi (not_mem_other hnd hu hu' hiu' hiu hgu))
      rw [read_eq C hc, hρ, ← hrun, ih i (htop.1 g x hx i hi) u' hu' hiu']

/-- On a correct set of gates, every gate of it computes a value. -/
theorem runIn_some (htop : C.Topological) {S : List ℕ} {val : ℕ → Option Bool}
    (hlen : ∀ g ∈ S, g < C.gates.length) (hc : C.Correct val S) : ∀ g ∈ S, ∃ b, C.runIn val S g = some b := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    intro hgS
    have hgl : g < C.gates.length := hlen g hgS
    have hx : C.gates[g]? = some C.gates[g] := List.getElem?_eq_getElem hgl
    rw [runIn_eq, hx, Option.bind_some]
    by_cases hin : C.gates[g] = .input
    · obtain ⟨b, hb, -⟩ := (hc g hgS).2 (boundary_input C (hin ▸ hx))
      exact ⟨b, by rw [hin]; simp [BGate.val, hb]⟩
    · apply Option.isSome_iff_exists.1
      apply val_isSome _ hin
      intro i hi
      have hig : i < g := htop.1 g _ hx i hi
      by_cases hiS : i ∈ S
      · obtain ⟨b, hb⟩ := ih i hig hiS
        simp [hiS, hig, hb]
      · simp only [hiS, ite_false]
        exact Option.ne_none_iff_isSome.1 ((hc g hgS).1 _ hx i hi hiS)

/-- **A correct replay unit has an interior making every proof unit correct**: the values its gates compute. -/
theorem units_of_correct (htop : C.Topological) {r : RU} (hlen : ∀ g ∈ RU.gates r, g < C.gates.length)
    {val : ℕ → Option Bool} (hc : C.Correct val (RU.gates r)) :
    ∃ ι : ℕ → Bool, ∀ u ∈ r, C.Correct (RU.ref C r val ι) u := by
  have hD := runIn_some C htop hlen hc
  let ι : ℕ → Bool := fun g => (C.runIn val (RU.gates r) g).getD false
  have hE : ∀ i ∈ RU.gates r, RU.ref C r val ι i = C.runIn val (RU.gates r) i := by
    intro i hiS
    by_cases hb : C.Boundary (RU.gates r) i = true
    · obtain ⟨b, hv, hrun⟩ := (hc i hiS).2 hb
      rw [ref_boundary C hb, hv, hrun]
    · obtain ⟨b, hrun⟩ := hD i hiS
      rw [RU.ref, ite_eq_left ⟨hiS, by simpa using hb⟩, hrun]
      simp [ι, hrun]
  have hcross : ∀ i ∈ RU.gates r, C.read (RU.ref C r val ι) i = C.runIn val (RU.gates r) i := by
    intro i hiS
    by_cases hcst : ∃ b, C.gates[i]? = some (.const b)
    · obtain ⟨b, hb⟩ := hcst
      rw [read_const C hb, runIn_const C hb]
    · push Not at hcst
      rw [read_eq C hcst, hE i hiS]
  have hagree : ∀ g, ∀ u ∈ r, g ∈ u → C.runIn (RU.ref C r val ι) u g = C.runIn val (RU.gates r) g := by
    intro g
    induction g using Nat.strong_induction_on with
    | _ g ih =>
      intro u hu hgu
      exact runIn_step C htop (sub_gates hu) (ref_input C) fun x hx i hi =>
        ⟨fun hiu => ih i (htop.1 g x hx i hi) u hu hiu, fun _ hiS => hcross i hiS,
          fun hiS => read_congr C (ref_out C hiS)⟩
  refine ⟨ι, fun u hu g hgu => ⟨fun x hx i hi hiu => ?_, fun _ => ?_⟩⟩
  · by_cases hiS : i ∈ RU.gates r
    · obtain ⟨b, hb⟩ := hD i hiS
      rw [hcross i hiS, hb]
      simp
    · rw [read_congr C (ref_out C hiS)]
      exact (hc g (sub_gates hu g hgu)).1 x hx i hi hiS
  · obtain ⟨b, hb⟩ := hD g (sub_gates hu g hgu)
    exact ⟨b, by rw [hE g (sub_gates hu g hgu), hb], by rw [hagree g u hu hgu, hb]⟩

/-- **The core**: disjoint proof units have an interior making each correct exactly when the replay unit's set of gates
is correct on the committed values. -/
theorem core (htop : C.Topological) {r : RU} (hnd : (RU.gates r).Nodup)
    (hlen : ∀ g ∈ RU.gates r, g < C.gates.length) (val : ℕ → Option Bool) :
    (∃ ι : ℕ → Bool, ∀ u ∈ r, C.Correct (RU.ref C r val ι) u) ↔ C.Correct val (RU.gates r) := by
  refine ⟨fun ⟨ι, hcor⟩ g hgS => ?_, units_of_correct C htop hlen⟩
  obtain ⟨u, hu, hgu⟩ := List.mem_flatten.1 hgS
  have hc := hcor u hu g hgu
  refine ⟨fun x hx i hi hiS => ?_, fun hb => ?_⟩
  · have := hc.1 x hx i hi fun hiu => hiS (sub_gates hu i hiu)
    rwa [read_congr C (ref_out C hiS)] at this
  · obtain ⟨b, hρ, hrun⟩ := hc.2 (boundary_mono C (sub_gates hu) hb)
    refine ⟨b, by rwa [ref_boundary C hb] at hρ, ?_⟩
    rw [← runIn_units C htop hnd hcor g u hu hgu, hrun]

end Claims.Sampling.Boundary

open Claims.Sampling.Boundary in
/-- **The faulty set is the boundary form's** (`Claims.Sampling.FaultyBoundary`). -/
theorem Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary := by
  intro H Cm s P w v r htop hr hnd hlen
  simp only [Claims.Sampling.Faulty, Claims.Sampling.BoundaryFaulty, hr, Option.some.injEq, exists_eq_left']
  exact not_congr (core s.prog htop hnd hlen _)
