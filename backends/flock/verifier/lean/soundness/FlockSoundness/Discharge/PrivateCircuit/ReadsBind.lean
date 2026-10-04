import FlockSoundness.Discharge.PrivateCircuit.ReadsDefs

/-!
# Hidden-position reads bind the registered store and table, under `cr/sha-512`

A unit's gadget opens a registered root at a hidden index inside its own circuit. Two openings of one index that both
check against one root and carry different values are an explicit SHA-512 collision (`read_binding`): the leaf scheme's
extractor when their hm96 leaves agree, else the registered reads' extractor on the two leaves and paths. So:

* `regCommit_binding`: the commitment of `n` values under one root binds (`Commit.Binding`), which `ran_registered` needs
  for the topology table.
* `reads_bind_le`: in any game, accepting while some read of a family (`Rd`, the reads an outcome makes: a drawn unit's
  inputs and outputs, or the values delivered to the verifier) checks against the registered root at an index where its
  value is not the registrant's has probability at most `q²/2^513`, under `SHA512CRStrict` for the finder that plays the
  game once and outputs the extractor's pair of the first such read and the registrant's opening (`readPick`). It covers
  the row store's reads and the topology table's.
-/

namespace FlockSoundness.Discharge.PrivateCircuit.Reads

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

variable {α : Type}

/-- **The Merkle-read binding**: two openings of index `i` that both check against the root `rt` and carry different
values give an explicit SHA-512 collision, `readPair`'s. -/
theorem read_binding (lf : ExplicitLeaf α) (rt : Flock.Registered.Port) (i : ℕ) (x y : ReadVal α)
    (hx : ReadOk lf rt i x) (hy : ReadOk lf rt i y) (hne : x.val ≠ y.val) : Collides (readPair lf rt i x y) := by
  sorry

/-- **The commitment of `n` values under one registered root binds**: two openings of one root to two different value
vectors give an explicit SHA-512 collision, `readPair` at the first index where they differ. -/
theorem regCommit_binding (lf : ExplicitLeaf α) (n : ℕ) : (regCommit lf n).Binding := by
  sorry

open Classical in
/-- **The read finder's output** at an outcome `b`: for a read `Rd b i x` at the registrant's index `i` that checks
against `rt` with a value other than the registrant's opening `wo i`'s, the extractor's pair of the two openings; `([], [])`
when there is none. -/
noncomputable def readPick {β Ix : Type} (lf : ExplicitLeaf α) (rt : Flock.Registered.Port) (idx : Ix → ℕ)
    (wo : Ix → ReadVal α) (Rd : β → Ix → ReadVal α → Prop) (b : β) : List UInt8 × List UInt8 :=
  if h : ∃ i x, Rd b i x ∧ ReadOk lf rt (idx i) x ∧ x.val ≠ (wo i).val then
    readPair lf rt (idx (Classical.choose h)) (Classical.choose (Classical.choose_spec h)) (wo (Classical.choose h))
  else ([], [])

/-- **The reads bind the registrant's openings, under `cr/sha-512`.** The registrant's opening `wo i` of each index
checks against `rt` (`hwo`). In any game `G` played by `σ`, an outcome making a read that checks against `rt` at an index
where its value is not the registrant's has probability at most `q²/2^513`, under `SHA512CRStrict` for the finder that
outputs `readPick`, at most `q` SHA-512 evaluations on every outcome. -/
theorem reads_bind_le {β Ix : Type} (G : Game β) (σ : Strategy G) (lf : ExplicitLeaf α) (rt : Flock.Registered.Port)
    (idx : Ix → ℕ) (wo : Ix → ReadVal α) (hwo : ∀ i, ReadOk lf rt (idx i) (wo i)) (Rd : β → Ix → ReadVal α → Prop)
    (cost : β → ℝ) (q : ℝ) (hcost : ∀ b, cost b ≤ q)
    (hCR : Assumptions.SHA512CRStrict H512 G σ (fun b => some (readPick lf rt idx wo Rd b)) cost q) :
    prob (fun b => ∃ i x, Rd b i x ∧ ReadOk lf rt (idx i) x ∧ x.val ≠ (wo i).val) G σ
      ≤ ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit.Reads
