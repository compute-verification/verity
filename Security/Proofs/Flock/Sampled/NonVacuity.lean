import Mathlib.Data.Fintype.BigOperators
import Proofs.Flock.Sampled.Statements

/-!
# F2's statements' hypotheses hold together

One instance for all of them: one padded index, one type, nothing in the values (`Ty₁`, `X₁`), a one-leaf root holding
the unit's record (`onePort`: its root is the climb of the record's leaf, so the opening holds without computing SHA-512),
and one session of one slot holding the unit (`batch₁`).

* `coverage_nonvacuous`, `slotBinding_nonvacuous`, `bridge_nonvacuous`: the hypotheses of `Coverage`, `SlotBinding` and
  `Bridge` at that instance, the draw being the one unit.
* `menu₀_shapeMenu`, `menu₀_room`: `menu₀`, one-slot sessions, one per drawn unit, each staged over the inner shape
  `m = 22, k_log = 6`, two claims. `ShapeMenu menu₀` given that V* stages it (`menu₀.staged`, which `#guard` checks by
  evaluation, not by proof: building V*'s algebra is beyond the kernel's evaluation), and `Menu.Room` by proof;
  `menuComplete_nonvacuous`, `MenuComplete`'s hypotheses there, given the same.
* `composition_nonvacuous`: `Composition`'s hypotheses at that instance, under a one-point law and a session that
  verifies at once, with an instance of F1's interface (`I₁`) whose break fires at no draw.
-/

namespace FlockSampled

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

/-- One type, whose relation holds everywhere and whose matrix has no nonzeros. -/
def Ty₁ : Types Unit Unit := { holds := fun _ _ => True, nnz := fun _ => 0 }

/-- The one unit's record, everywhere. -/
def X₁ : Trace 1 Unit Unit := fun _ => some ((), ())

/-- `k` sessions of one slot each, session `b` holding unit `b`. -/
def batch₁ (k : ℕ) : Batching k Unit Unit where
  B := k
  shape := fun _ => ⟨1, 0⟩
  typ := fun _ => ()
  idx := fun b _ => some b
  val := fun _ _ => ()

theorem batch₁_cover (k : ℕ) : CoverCheck Finset.univ (batch₁ k) where
  drawn := fun _ _ _ => Finset.mem_univ _
  once := by
    rintro ⟨b, j⟩ ⟨b', j'⟩ i h h'
    have e : b = b' := Option.some.inj (h.trans h'.symm : some b = some b')
    subst e
    have hj : (j : ℕ) < 1 := j.isLt
    have hj' : (j' : ℕ) < 1 := j'.isLt
    have e' : j = j' := Fin.ext (by omega)
    subst e'
    rfl
  count := by
    have hall : ∀ p ∈ (Finset.univ : Finset (batch₁ k).Pos), ((batch₁ k).held p).isSome = true := fun _ _ => rfl
    first
      | (rw [Finset.filter_true_of_mem hall, Finset.card_univ, Finset.card_univ, Fintype.card_sigma]
         simp [batch₁]
         done)
      | (simp [Batching.held, batch₁, Fintype.card_sigma]
         done)

/-- A one-leaf root whose leaf is the record `bc`'s. -/
def onePort (bc : ByteArray) : Flock.Registered.Port where
  name := ""
  value := ""
  root := Flock.Registered.rowLeaf ByteArray.empty 0 (Flock.Hm96.Default512.treeLeaf bc)
  leaves := 1
  words := none
  wordBits := 0
  domain := ByteArray.empty
  reads := #[]

theorem climb_one (dom leaf path : ByteArray) (i : ℕ) :
    Flock.Registered.climb dom leaf i (Flock.log2ceil 1) path = leaf := by
  first
    | rfl
    | simp [Flock.Registered.climb, Flock.Registered.sibs, Flock.Registered.climbL, Flock.log2ceil]

theorem onePort_opens (bc : ByteArray) : Flock.Registered.opens (onePort bc) bc 0 ByteArray.empty = true := by
  simp only [Flock.Registered.opens, Flock.Registered.opensLeaf, Bool.and_eq_true, decide_eq_true_eq]
  exact ⟨⟨Nat.zero_lt_one, rfl⟩, climb_one _ _ _ _⟩

/-- The commitment: the one-leaf root, every record empty, every unit at position 0. -/
def C₁ : Commit 1 Unit Unit := { rt := onePort ByteArray.empty, enc := fun _ _ => ByteArray.empty, pos := fun _ => 0 }

/-- The committed trace's openings. -/
def path₀₁ : Fin 1 → ByteArray := fun _ => ByteArray.empty

/-- One session of one slot holding the unit, and the slot's opening. -/
def h₁ : Hidden 1 Unit Unit := { β := batch₁ 1, path := fun _ => ByteArray.empty }

theorem C₁_injective : C₁.Injective := fun _ _ _ _ _ => ⟨rfl, rfl⟩

theorem C₁_committed : C₁.Committed X₁ path₀₁ where
  opens := fun _ _ _ _ => onePort_opens _
  full := fun _ _ _ _ _ => rfl

theorem h₁_opens : C₁.SlotOpens h₁ := fun _ _ _ => onePort_opens _

theorem h₁_bound : Bound X₁ h₁.β := fun _ _ _ => rfl

theorem h₁_proves : Proves Ty₁ C₁ Finset.univ h₁ where
  cover := batch₁_cover 1
  opens := h₁_opens
  fits := fun _ => Nat.zero_le _
  holds := fun _ _ _ => trivial

/-- **`Coverage`'s hypotheses hold together**, at a nonempty draw. -/
theorem coverage_nonvacuous :
    (Finset.univ : Finset (Fin 1)).Nonempty ∧ CoverCheck Finset.univ (batch₁ 1) ∧ Bound X₁ (batch₁ 1) :=
  ⟨Finset.univ_nonempty, batch₁_cover 1, h₁_bound⟩

/-- **`SlotBinding`'s hypotheses hold together**, at a slot that holds a unit. -/
theorem slotBinding_nonvacuous :
    C₁.Injective ∧ C₁.Committed X₁ path₀₁ ∧ C₁.SlotOpens h₁ ∧ ∃ p i, h₁.β.held p = some i :=
  ⟨C₁_injective, C₁_committed, h₁_opens, ⟨⟨⟨0, Nat.zero_lt_one⟩, ⟨0, Nat.zero_lt_one⟩⟩, 0, rfl⟩⟩

/-- **`Bridge`'s hypotheses hold together**, at a nonempty draw. -/
theorem bridge_nonvacuous :
    (Finset.univ : Finset (Fin 1)).Nonempty ∧ C₁.Injective ∧ C₁.Committed X₁ path₀₁ ∧ Proves Ty₁ C₁ Finset.univ h₁ :=
  ⟨Finset.univ_nonempty, C₁_injective, C₁_committed, h₁_proves⟩

/-- One-slot sessions of no nonzeros, one per drawn unit, over the inner shape `m = 22, k_log = 6`, two claims. -/
def menu₀ : Menu where
  shapes := [⟨1, 0⟩]
  inner := fun _ => ⟨22, 6, 2⟩
  padded := fun k => (List.finRange k).map fun _ => ⟨1, 0⟩

#guard menu₀.staged

theorem isOk_iff {ε α : Type} (x : Except ε α) : isOk x = true ↔ ∃ a, x = .ok a := by
  cases x <;> simp [isOk]

theorem Menu.staged_ok {M : Menu} (h : M.staged = true) : ∀ s ∈ M.shapes, ∃ vs, M.vstar s = .ok vs :=
  fun s hs => (isOk_iff _).1 (List.all_eq_true.1 h s hs)

/-- **`menu₀` is a shape menu**, given that V* stages it (`#guard menu₀.staged`, by evaluation). -/
theorem menu₀_shapeMenu (h : menu₀.staged = true) : ShapeMenu menu₀ where
  staged := Menu.staged_ok h
  padded := by
    intro k s hs
    obtain ⟨_, _, e⟩ := List.mem_map.1 hs
    cases e
    exact List.mem_singleton_self _

/-- **`menu₀`'s padded sessions have room**, for `Ty₁`. -/
theorem menu₀_room : menu₀.Room Ty₁ :=
  fun k _ => ⟨batch₁ k, rfl, batch₁_cover k, fun _ _ _ => rfl, fun _ => Nat.zero_le _⟩

/-- **`MenuComplete`'s hypotheses hold together**, at `menu₀` given that V* stages it, `Ty₁`, and a nonempty draw whose
units `X₁` holds. -/
theorem menuComplete_nonvacuous (h : menu₀.staged = true) :
    ShapeMenu menu₀ ∧ menu₀.Room Ty₁ ∧ (Finset.univ : Finset (Fin 1)).Nonempty ∧
      ∀ i ∈ (Finset.univ : Finset (Fin 1)), (X₁ i).isSome = true :=
  ⟨menu₀_shapeMenu h, menu₀_room, Finset.univ_nonempty, fun _ _ => rfl⟩

/-- The one-point law over the one unit. -/
def L₁ : Law 1 := { Ω := Unit, draw := fun _ => Finset.univ }

/-- Sessions that end at once. -/
def session₁ : Finset (Fin 1) → Unit → Game Unit := fun _ _ => .ret ()

/-- **An instance of F1's interface**: the sessions always verify, hold `h₁`, and break only away from the one draw. -/
noncomputable def I₁ : Recursion session₁ Ty₁ (fun _ => C₁) where
  verifies := fun _ => true
  hid := fun _ _ _ => h₁
  Break := fun S _ _ => S ≠ Finset.univ
  ε := fun _ _ _ => 0
  sound := fun S _ _ => by
    by_cases hS : S = Finset.univ
    · subst hS
      refine Or.inr (le_of_le_of_eq (prob_mono (q := fun _ => False) ?_ _ _) (prob_false _ _))
      intro _ h
      exact h.2 h₁_proves
    · exact Or.inl hS

/-- **`Composition`'s hypotheses hold together**: at the one-point law, `session₁` and `I₁`, the records determine
their contents, `X₁` is the trace committed, and the break of F1's interface fires at no draw of any prover. -/
theorem composition_nonvacuous :
    (∀ _ : Unit, C₁.Injective) ∧ (∀ _ : Unit, C₁.Committed X₁ path₀₁) ∧
      ∀ σ : Strategy (sampledAudit L₁ session₁), ¬ ∃ ω, I₁.Break (L₁.draw ω) (regOf σ) (contOf σ ω) :=
  ⟨fun _ => C₁_injective, fun _ => C₁_committed, fun _ ⟨_, h⟩ => h rfl⟩

end FlockSampled
