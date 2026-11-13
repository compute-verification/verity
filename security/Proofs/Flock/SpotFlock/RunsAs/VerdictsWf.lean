module

public import Proofs.Flock.SpotFlock.RunsAs.Steps
public import Spot.OneStageZk

/-!
# The verdicts the per-member form's run returns are JSON objects

`Spot.OneStageZk.membersRun` returns a run whose verdicts (`Run.out`) are pushed one a session by `push`, each
`verdictJson`'s: a pair whose value is built by `Json.mkObj`. So every verdict of an accepted `membersRun` is a JSON
object over a well-formed map (`OutWf`, `membersRun_wf`), which a reading of a verdict's keys after other keys are set
needs (`Std.TreeMap.Raw.getElem?_insert`).

The walk is `membersRun`, `runMembers`, `stepMember`, `runFrom`, `step`, `push`, `verdictJson`, each keeping `OutWf`.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- Every verdict's value is a JSON object over a well-formed map. -/
def OutWf (out : Array (String × Json)) : Prop :=
  ∀ x ∈ out, ∃ m : Std.TreeMap.Raw String Json compare, x.2 = .obj m ∧ m.WF

private theorem bind_iff {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

private theorem bind_ok' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := bind_iff.1 h

private theorem throw_ne' {α : Type} (e : String) (a : α) : (throw e : Flock.V α) ≠ .ok a := fun h => by cases h

private theorem mapError_ok' {ε α : Type} {x : Except ε α} {f : ε → ε} {a : α} (h : x.mapError f = .ok a) :
    x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

theorem OutWf.empty : OutWf #[] := fun x hx => by simp at hx

theorem OutWf.push {out : Array (String × Json)} (h : OutWf out) {x : String × Json}
    (hx : ∃ m : Std.TreeMap.Raw String Json compare, x.2 = .obj m ∧ m.WF) : OutWf (out.push x) := by
  intro y hy
  rcases Array.mem_push.1 hy with hy | rfl
  · exact h y hy
  · exact hx

/-- A session's verdict, as `verdictJson` writes it, is an object `Json.mkObj` builds. -/
theorem verdictJson_wf (rd : Spot.OneStageZk.Reads) (programSha : Option String) (s : Spot.OneStageZk.RunFiles)
    (st : Spot.OneStageZk.SetUp) (ok : Bool) (why : String) (units : Array Nat) :
    ∃ m : Std.TreeMap.Raw String Json compare,
      (Spot.OneStageZk.verdictJson rd programSha s st ok why units).2 = .obj m ∧ m.WF := by
  unfold Spot.OneStageZk.verdictJson
  rcases st with e | ⟨a, b, p, o⟩
  · exact mkObj_wf _
  · exact mkObj_wf _

theorem push_wf {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {programSha : Option String} {want : Json}
    {r r' : Spot.OneStageZk.Run} {s : Spot.OneStageZk.RunFiles} {record : String} {st : Spot.OneStageZk.SetUp}
    {coinKey : Option String} {verdict : Flock.V Unit}
    (h : Spot.OneStageZk.push I rd programSha want r s record st coinKey verdict = .ok r') (hr : OutWf r.out) :
    OutWf r'.out := by
  unfold Spot.OneStageZk.push at h
  try dsimp only at h
  split at h
  · obtain rfl := Except.ok.inj h
    exact hr.push (verdictJson_wf ..)
  · split at h
    · exact absurd h (throw_ne' _ _)
    obtain ⟨units, -, h⟩ := bind_ok' h
    obtain rfl := Except.ok.inj h
    exact hr.push (verdictJson_wf ..)

theorem step_wf {I : Flock.Witness.Inputs} {rd : Spot.OneStageZk.Reads} {bs : Core.Blobs}
    {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {r r' : Spot.OneStageZk.Run} {s : Spot.OneStageZk.RunFiles}
    (h : Spot.OneStageZk.step I rd bs scope ksFiles population programSha want r s = .ok r') (hr : OutWf r.out) :
    OutWf r'.out := by
  unfold Spot.OneStageZk.step at h
  obtain ⟨b, -, h⟩ := bind_ok' h
  split at h
  · try dsimp only at h
    obtain ⟨⟨⟩, -, h⟩ := bind_ok' h
    exact push_wf h hr
  · exact absurd h (throw_ne' _ _)

theorem foldl_wf {a : Spot.OneStageZk.StepAt} {bs : Core.Blobs} :
    ∀ (l : List Spot.OneStageZk.RunFiles) (r r' : Spot.OneStageZk.Run),
      l.foldlM (fun r s => a.step bs r s) r = .ok r' → OutWf r.out → OutWf r'.out
  | [], r, r', h, hr => by
    obtain rfl : r = r' := Except.ok.inj h
    exact hr
  | s :: l, r, r', h, hr => by
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := bind_ok' h
    have h₁' : Spot.OneStageZk.step a.I a.rd bs a.scope a.ksFiles a.population a.programSha a.want r s =
        .ok r₁ := h₁
    exact foldl_wf l r₁ r' h (step_wf h₁' hr)

theorem stepMember_wf {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs}
    {r r' : Spot.OneStageZk.Run} {z : Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member}
    (h : Spot.OneStageZk.stepMember st bs r z = .ok r') (hr : OutWf r.out) : OutWf r'.out := by
  unfold Spot.OneStageZk.stepMember at h
  have h := mapError_ok' h
  unfold Spot.OneStageZk.runFrom at h
  obtain ⟨a, -, h⟩ := bind_ok' h
  rw [← Array.foldlM_toList] at h
  exact foldl_wf _ r r' h hr

theorem foldlMembers_wf {st : Spot.OneStageZk.Member → Flock.V Spot.OneStageZk.StepAt} {bs : Core.Blobs} :
    ∀ (l : List (Spot.OneStageZk.StatementFiles × Spot.OneStageZk.Member)) (r r' : Spot.OneStageZk.Run),
      l.foldlM (Spot.OneStageZk.stepMember st bs) r = .ok r' → OutWf r.out → OutWf r'.out
  | [], r, r', h, hr => by
    obtain rfl : r = r' := Except.ok.inj h
    exact hr
  | z :: l, r, r', h, hr => by
    rw [List.foldlM_cons] at h
    obtain ⟨r₁, h₁, h⟩ := bind_ok' h
    exact foldlMembers_wf l r₁ r' h (stepMember_wf h₁ hr)

theorem runMembers_wf {slice : Spot.OneStageZk.Member → Json} {ms : Array Spot.OneStageZk.Member}
    {sts : Array Spot.OneStageZk.StatementFiles} {want : Array Nat} {bs : Core.Blobs} {r : Spot.OneStageZk.Run}
    (h : Spot.OneStageZk.runMembers slice ms sts want bs = .ok r) : OutWf r.out := by
  unfold Spot.OneStageZk.runMembers at h
  obtain ⟨held, -, h⟩ := bind_ok' h
  obtain ⟨_, -, h⟩ := bind_ok' h
  obtain ⟨r₁, h₁, h⟩ := bind_ok' h
  obtain ⟨_, -, h⟩ := bind_ok' h
  obtain rfl : r₁ = r := Except.ok.inj h
  rw [← Array.foldlM_toList] at h₁
  exact foldlMembers_wf _ _ _ h₁ OutWf.empty

/-- **The verdicts of an accepted `membersRun` are JSON objects**, each over a well-formed map. -/
theorem membersRun_wf {p : Spot.OneStageZk.Params} {ms : Array Spot.OneStageZk.Member}
    {c : Spot.OneStage.Commitment} {e : Spot.OneStageZk.Evidence} {sts : Array Spot.OneStageZk.StatementFiles}
    {bs : Core.Blobs} {r : Spot.OneStageZk.Run} (h : Spot.OneStageZk.membersRun p ms c e sts bs = .ok r) :
    OutWf r.out := by
  unfold Spot.OneStageZk.membersRun at h
  obtain ⟨_, -, h⟩ := bind_ok' h
  obtain ⟨_, -, h⟩ := bind_ok' h
  exact runMembers_wf h

end SpotFlock.RunsAs
