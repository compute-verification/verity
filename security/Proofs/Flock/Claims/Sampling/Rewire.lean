module

public import Proofs.Flock.Claims.Sampling.ProgReads

/-!
# A program with its reads sent through a map

`C.rewire ω` is `C` with every gate reading wire `ω i` where it read `i`. Where `ω` sends an input gate one proof block
reads to the gate another proof block writes, the rewired program is the one in which that value is one gate, which
both blocks need.

* `BGate.reads_rewire`, `BGate.val_rewire`, `Prog.rewire_gates`, `Prog.gate_rewire`: a rewired gate reads the images
  of the wires it read, and its value on values `rd` is the gate's on `rd` read through `ω`.
* `Prog.needs_rewire_of_read`: a set of gates one of which read `i` needs `ω i` at the rewired program, where `ω i` is
  outside the set and no constant.
* `Prog.boundary_rewire_of_read`, `Prog.needs_rewire_of_written`: at the rewired program `ω i` is on the boundary of
  every set that leaves out a gate that read `i`, so a set that holds `ω i` needs it.
* `Prog.needs_rewire_both`: so a set that holds `ω i` and a set one of whose gates read `i` both need `ω i`.
-/

@[expose] public section

namespace Claims.Sampling

/-- **Gate `x`, every wire it reads sent through `ω`.** -/
def BGate.rewire (ω : ℕ → ℕ) : BGate → BGate
  | .input => .input
  | .const c => .const c
  | .not i => .not (ω i)
  | .and ws => .and (ws.map ω)
  | .xor ws => .xor (ws.map ω)

/-- **`C` rewired by `ω`**: `C`'s gates, each reading wire `ω i` where it read `i`, and `C`'s returned gates. -/
def Prog.rewire (C : Prog) (ω : ℕ → ℕ) : Prog := ⟨C.gates.map (BGate.rewire ω), C.outs⟩

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

/-- **A rewired gate's value**: the gate's, on the values read through `ω`. -/
theorem BGate.val_rewire (ω : ℕ → ℕ) (x : BGate) (rd : ℕ → Option Bool) (own : Option Bool) :
    (x.rewire ω).val rd own = x.val (fun i => rd (ω i)) own := by
  cases x <;> simp only [BGate.rewire, BGate.val, mapM_map_rd]

theorem Prog.rewire_gates (C : Prog) (ω : ℕ → ℕ) (g : ℕ) :
    (C.rewire ω).gates[g]? = C.gates[g]?.map (BGate.rewire ω) := by
  simp [Prog.rewire]

/-- **A gate of the rewired program** is a gate of `C` rewired, and evaluates on values `v` as that gate does on `v`
read through `ω`. -/
theorem Prog.gate_rewire {C : Prog} {ω : ℕ → ℕ} {g : ℕ} {y : BGate} (hy : (C.rewire ω).gates[g]? = some y) :
    ∃ x, C.gates[g]? = some x ∧ y = x.rewire ω ∧
      ∀ (v : ℕ → Bool) (own : Option Bool),
        y.val (fun i => some (v i)) own = x.val (fun i => some (v (ω i))) own := by
  rw [Prog.rewire_gates] at hy
  obtain ⟨x, hx, rfl⟩ := Option.map_eq_some_iff.1 hy
  exact ⟨x, hx, rfl, fun v own => BGate.val_rewire ω x (fun i => some (v i)) own⟩

/-- **The reader needs the wire's gate**: a set of gates `u`, one of whose gates read `i` in `C`, needs `ω i` at the
rewired program, where `ω i` is outside `u` and no constant. -/
theorem Prog.needs_rewire_of_read {C : Prog} {ω : ℕ → ℕ} {u : List ℕ} {g' i : ℕ} {x : BGate} (hg' : g' ∈ u)
    (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) (hout : ω i ∉ u)
    (hnc : ∀ c, C.gates[ω i]? ≠ some (.const c)) : (C.rewire ω).Needs u (ω i) := by
  have hx' : (C.rewire ω).gates[g']? = some (x.rewire ω) := by
    rw [Prog.rewire_gates, hx, Option.map_some]
  have hi' : ω i ∈ (x.rewire ω).reads := by
    rw [BGate.reads_rewire]
    exact List.mem_map.2 ⟨i, hi, rfl⟩
  unfold Prog.Needs
  refine Or.inr ⟨⟨g', hg', x.rewire ω, hx', hi'⟩, hout, fun c hc => ?_⟩
  rw [Prog.rewire_gates] at hc
  obtain ⟨y, hy, hyc⟩ := Option.map_eq_some_iff.1 hc
  rw [BGate.rewire_eq_const.1 hyc] at hy
  exact hnc c hy

/-- **The wire's gate is on the boundary of a set that leaves out its reader**: at the rewired program, `ω i` is
committed as a set `v` sees it, where a gate outside `v` read `i` in `C`. -/
theorem Prog.boundary_rewire_of_read {C : Prog} {ω : ℕ → ℕ} (v : List ℕ) {g' i : ℕ} {x : BGate} (hg' : g' ∉ v)
    (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) : (C.rewire ω).Boundary v (ω i) = true := by
  have hx' : (C.rewire ω).gates[g']? = some (x.rewire ω) := by
    rw [Prog.rewire_gates, hx, Option.map_some]
  have hi' : ω i ∈ (x.rewire ω).reads := by
    rw [BGate.reads_rewire]
    exact List.mem_map.2 ⟨i, hi, rfl⟩
  exact Prog.boundary_of_read hg' (List.getElem?_eq_some_iff.1 hx').1 hx' hi'

/-- **The writer needs the wire's gate**: a set of gates `v` that holds `ω i` needs it at the rewired program, where
a gate outside `v` read `i` in `C`. -/
theorem Prog.needs_rewire_of_written {C : Prog} {ω : ℕ → ℕ} {v : List ℕ} {g' i : ℕ} {x : BGate} (hw : ω i ∈ v)
    (hg' : g' ∉ v) (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) : (C.rewire ω).Needs v (ω i) := by
  unfold Prog.Needs
  exact Or.inl ⟨hw, Prog.boundary_rewire_of_read v hg' hx hi⟩

/-- **One wire, two sets that need its gate**: where a gate of `u` read `i` in `C` and `ω i` is a gate of `v`, outside
`u` and no constant, and the reading gate is outside `v`, both `u` and `v` need `ω i` at the rewired program. -/
theorem Prog.needs_rewire_both {C : Prog} {ω : ℕ → ℕ} {u v : List ℕ} {g' i : ℕ} {x : BGate} (hg' : g' ∈ u)
    (hx : C.gates[g']? = some x) (hi : i ∈ x.reads) (hw : ω i ∈ v) (hwu : ω i ∉ u) (hg'v : g' ∉ v)
    (hnc : ∀ c, C.gates[ω i]? ≠ some (.const c)) :
    (C.rewire ω).Needs u (ω i) ∧ (C.rewire ω).Needs v (ω i) :=
  ⟨Prog.needs_rewire_of_read hg' hx hi hwu hnc, Prog.needs_rewire_of_written hw hg'v hx hi⟩

end Claims.Sampling
