import Definitions.Claims.Sampling

/-!
# The sampling block's faulty set is the boundary form's

`Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary`. A proof unit's check reads its interior values
nowhere: a value it reads from outside it is committed, and a gate of it computes from its own wires as it computes them
and from an input's committed value, which is on its boundary (`runIn_refIn`). So it is correct under some interior
exactly when it is correct on the committed values, under any interior (`core`).
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

theorem val_congr (x : BGate) {rd rd' : ℕ → Option Bool} {own own' : Option Bool} (h : ∀ i ∈ x.reads, rd i = rd' i)
    (ho : x = .input → own = own') : x.val rd own = x.val rd' own' := by
  cases x with
  | input => exact ho rfl
  | const b => rfl
  | not i => simp only [BGate.val, h i (by simp [BGate.reads])]
  | and ws => simp only [BGate.val, mapM_congr' ws h]
  | xor ws => simp only [BGate.val, mapM_congr' ws h]

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

theorem boundary_input {u : List ℕ} {g : ℕ} (h : C.gates[g]? = some .input) : C.Boundary u g = true := by
  simp [Prog.Boundary, h]

theorem ref_out {u : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : i ∉ u) :
    C.refIn u val ι i = val i := by
  rw [Prog.refIn, ite_eq_right fun h' => h h'.1]

theorem ref_boundary {u : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : C.Boundary u i = true) :
    C.refIn u val ι i = val i := by
  rw [Prog.refIn, ite_eq_right fun h' => by simp [h] at h']

theorem ref_input {u : List ℕ} {val : ℕ → Option Bool} {ι : ℕ → Bool} {i : ℕ} (h : C.gates[i]? = some .input) :
    C.refIn u val ι i = val i :=
  ref_boundary C (boundary_input C h)

/-- **A proof unit computes the same whatever its interior**: a gate of `u` reads a wire of `u` as `u` computes it, a
wire outside `u` as committed, and an input's own value, on `u`'s boundary, as committed. -/
theorem runIn_refIn (u : List ℕ) (val : ℕ → Option Bool) (ι : ℕ → Bool) :
    ∀ g, C.runIn (C.refIn u val ι) u g = C.runIn val u g := by
  intro g
  induction g using Nat.strong_induction_on with
  | _ g ih =>
    rw [runIn_eq, runIn_eq]
    cases hx : C.gates[g]? with
    | none => rfl
    | some x =>
      simp only [Option.bind_some]
      refine val_congr x (fun i _ => ?_) (fun h => ref_input C (by rw [hx, h]))
      by_cases hiu : i ∈ u
      · rw [if_pos hiu, if_pos hiu]
        by_cases hig : i < g
        · rw [if_pos hig, if_pos hig, ih i hig]
        · rw [if_neg hig, if_neg hig]
      · rw [if_neg hiu, if_neg hiu]
        exact read_congr C (ref_out C hiu)

/-- **The core**: a proof unit is correct under an interior exactly when it is correct on the committed values. -/
theorem core (u : List ℕ) (val : ℕ → Option Bool) (ι : ℕ → Bool) :
    C.Correct (C.refIn u val ι) u ↔ C.Correct val u := by
  unfold Prog.Correct
  refine forall₂_congr fun g _ => and_congr (forall₂_congr fun x _ => forall₂_congr fun i _ => ?_) ?_
  · exact imp_congr_right fun hiu => by rw [read_congr C (ref_out C hiu)]
  · exact imp_congr_right fun hb => by rw [ref_boundary C hb, runIn_refIn]

end Claims.Sampling.Boundary

open Claims.Sampling.Boundary in
/-- **The faulty set is the boundary form's** (`Claims.Sampling.FaultyBoundary`). -/
theorem Claims.Sampling.faultyBoundary : Claims.Sampling.FaultyBoundary := by
  intro H Cm s P w v u _ hu _ _
  simp only [Claims.Sampling.Faulty, Claims.Sampling.BoundaryFaulty, hu, Option.some.injEq, exists_eq_left']
  exact not_congr ⟨fun ⟨ι, h⟩ => (core s.prog u _ ι).1 h, fun h => ⟨fun _ => false, (core s.prog u _ _).2 h⟩⟩
