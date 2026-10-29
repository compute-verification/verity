import Proofs.Flock.Bridge.Defs
import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-!
# The registration's population is the game's units, in order (`Bridge.PopMatches`)

`Bridge.PopMatches desc q n`, a decidable check on the files, as `Bridge.RowsOk` is: the descriptor bytes `desc` parse
(`SampledProofs.Program.parse`), the layout the deployed verifier evaluates on them under the registration's query `q`
(`SampledProofs.Slots.layout`, which reads the query as JSON, `Reg.query`) is defined, and its population, the layout's
first class, is `n` replay units with slot `u` unit `u`. It says nothing of the units' gates: that slot `u`'s gates are
unit `u`'s lowered gates needs the descriptor-to-circuit-file lowering, which is not yet stated.

* `popOf l`: a layout's population, its first class's replay units in slot order.
* `popMatches_iff`: `PopMatches` is that the parse and the layout succeed and the population's units are
  `0, …, n - 1`, in order.
* `popOf_size`, `popOf_unit`: at `PopMatches`, the population has `n` slots, and slot `u` is unit `u`.
* `popCount Pt`: the number of replay units in a claims partition's first class, its population.
* `popCount_slotPartition`: the claims' partition of a layout (`slotPartition`) has `popOf`'s slots as its population.
* `popCount_partOf`: at `PopMatches`, that population has as many slots as the game's partition `partOf P`'s population
  (`popCount_partOf_eq`): N = n.
* `popCount_regMatches`: where `RegMatches` names a statement's registration and `PopMatches` holds at the statement's
  descriptor and the registration's query, the statement's partition has `popCount (partOf P)` population slots.
-/

namespace Flock.Bridge

open Properties.SampledProofs

/-- **A layout's population**: its first class's replay units, in slot order. -/
def popOf (l : SampledProofs.Slots.Layout) : List SampledProofs.Slots.RU :=
  ((l.classes.toList.headD #[]).toList.map Array.toList).flatten

/-- `PopMatches`'s check. -/
def popCheck (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Bool :=
  match SampledProofs.Program.parse ⟨desc.toArray⟩ with
  | .error _ => false
  | .ok p =>
    match SampledProofs.Slots.layout p q with
    | .error _ => false
    | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)

/-- **The registration's population is `n` units, in order**: `desc` parses, its layout under the query `q` is
defined, and the layout's population is `n` replay units with slot `u` unit `u`. -/
@[reducible] def PopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Prop := popCheck desc q n = true

instance instDecidablePopMatches (desc : Claims.Msg) (q : Lean.Json) (n : ℕ) : Decidable (PopMatches desc q n) :=
  inferInstanceAs (Decidable (_ = true))

theorem popMatches_iff {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} :
    PopMatches desc q n ↔ ∃ p l, SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p ∧
      SampledProofs.Slots.layout p q = .ok l ∧ (popOf l).map (·.unit) = (List.range n).map some := by
  unfold PopMatches popCheck
  constructor
  · intro h
    split at h
    · exact absurd h Bool.false_ne_true
    · rename_i p hp
      split at h
      · exact absurd h Bool.false_ne_true
      · rename_i l hl
        exact ⟨p, l, hp, hl, of_decide_eq_true h⟩
  · rintro ⟨p, l, hp, hl, h⟩
    rw [hp]
    show (match SampledProofs.Slots.layout p q with
      | .error _ => false
      | .ok l => decide ((popOf l).map (·.unit) = (List.range n).map some)) = true
    rw [hl]
    exact decide_eq_true h

/-- The population's units at the layout `PopMatches` reads. -/
theorem popOf_units {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    {l : SampledProofs.Slots.Layout} (h : PopMatches desc q n)
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : SampledProofs.Slots.layout p q = .ok l) :
    (popOf l).map (·.unit) = (List.range n).map some := by
  obtain ⟨p', l', hp', hl', hu⟩ := popMatches_iff.1 h
  rw [hp] at hp'
  cases hp'
  rw [hl] at hl'
  cases hl'
  exact hu

/-- **N = n at the layout**: at `PopMatches`, the population has `n` slots. -/
theorem popOf_size {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    {l : SampledProofs.Slots.Layout} (h : PopMatches desc q n)
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : SampledProofs.Slots.layout p q = .ok l) :
    (popOf l).length = n := by
  have := congrArg List.length (popOf_units h hp hl)
  simpa only [List.length_map, List.length_range] using this

/-- **Slot `u` is unit `u`**: at `PopMatches`, the population's replay unit at slot `u` is the population's unit `u`. -/
theorem popOf_unit {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    {l : SampledProofs.Slots.Layout} (h : PopMatches desc q n)
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : SampledProofs.Slots.layout p q = .ok l)
    (u : ℕ) (hu : u < (popOf l).length) : ((popOf l)[u]'hu).unit = some u := by
  have hu' : u < ((popOf l).map (·.unit)).length := by rw [List.length_map]; exact hu
  have := List.getElem_of_eq (popOf_units h hp hl) hu'
  simpa only [List.getElem_map, List.getElem_range] using this

/-- **The population's slot count** of a claims partition: the replay units of its first class. -/
def popCount (Pt : Claims.Sampling.Partition) : ℕ := (Pt.classes.headD []).flatten.length

/-- The claims' partition of a layout has the layout's population as its population. -/
theorem popCount_slotPartition (l : SampledProofs.Slots.Layout) : popCount (slotPartition l) = (popOf l).length := by
  rcases l with ⟨⟨cl⟩⟩
  cases cl with
  | nil => rfl
  | cons c cs => simp [popCount, slotPartition, popOf, List.length_flatten, Function.comp_def]

/-- `partOf P`'s population has one slot per unit (the input class comes after it). -/
theorem popCount_partOf_eq {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n) :
    popCount (partOf P) = n := by
  simp [popCount, partOf]

/-- **N = n**: at `PopMatches`, the claims' partition of the deployed layout has as many population slots as the
game's partition `partOf P` has population slots. -/
theorem popCount_partOf {desc : Claims.Msg} {q : Lean.Json} {n : ℕ} {p : SampledProofs.Program}
    {l : SampledProofs.Slots.Layout} (h : PopMatches desc q n)
    (hp : SampledProofs.Program.parse ⟨desc.toArray⟩ = .ok p) (hl : SampledProofs.Slots.layout p q = .ok l)
    {C : FlockSoundness.Audit.Circuit} (P : FlockSoundness.Audit.Partition C n) :
    popCount (slotPartition l) = popCount (partOf P) := by
  rw [popCount_slotPartition, popOf_size h hp hl, popCount_partOf_eq]

/-- **N = n at the registration**: where `RegMatches` names the statement's registration, and `PopMatches` holds at
the statement's descriptor and that registration's query, the statement's partition has as many population slots as
`partOf P` has population slots. -/
theorem popCount_regMatches {c : SamplingDeploy.Code} {reg : Claims.Msg → Option SamplingDeploy.Reg}
    (hreg : SamplingDeploy.RegMatches c reg) {x : Claims.Msg} {s : Claims.Sampling.AuditStatement}
    {Pt : Claims.Sampling.Partition} (hx : c.parse x = some s) (hP : c.partition s = some Pt) {n : ℕ}
    (hpop : ∀ g, reg x = some g → PopMatches s.desc g.query n)
    {C : FlockSoundness.Audit.Circuit} (P : FlockSoundness.Audit.Partition C n) :
    popCount Pt = popCount (partOf P) := by
  obtain ⟨g, l, hg, hd, hl, hPt, -⟩ := hreg x s Pt hx hP
  rw [hPt]
  exact popCount_partOf (hpop g hg) hd hl P

end Flock.Bridge
