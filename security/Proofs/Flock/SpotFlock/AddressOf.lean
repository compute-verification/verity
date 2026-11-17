module

public import Proofs.Flock.SpotFlock.AddressFacts

/-! The address map of a held code (`addressOf h`), read off what the verifier holds and nothing of a run, at which the
five facts are to be proved. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

def bitAtGate (i : ℕ) : Option (ℕ × Spot.Bits.RowPos) :=
  match (sourceBits h.ms).find? (·.1 == i) with
  | some s => some s.2
  | none => ((writtenBits h.ms).find? (·.1 == i)).map (·.2)

def placeAt (t : ℕ) (r : Spot.Bits.RowPos) : (w : ℕ) × Fin w :=
  if hb : r.bit < 16 * ((memberCircuit h t).ports[portNamed h t r.row]!).words then
    ⟨16 * ((memberCircuit h t).ports[portNamed h t r.row]!).words, ⟨r.bit, hb⟩⟩
  else ⟨1, 0⟩

def slotAt (t : ℕ) (r : Spot.Bits.RowPos) : Slot :=
  match heldLink h (h.names.getD t "") r.row with
  | some l =>
    { key := .port l.value
      schema := _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t r.row)
      pos := r.leaf / h.reg.ns.getD t 0 * ((heldRows h (h.names.getD t "") r.row).map (·.per)).getD 0 +
        ((heldRows h (h.names.getD t "") r.row).bind fun rows =>
          Spot.SecondStage.rowAt rows.runs (r.leaf % h.reg.ns.getD t 0)).getD 0 }
  | none =>
    { key := .interior (h.names.getD t "") r.row
      schema := _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t r.row)
      pos := r.leaf % h.reg.ns.getD t 0
      stride := h.reg.ns.getD t 0
      sized := true }

/-- **The address no fact is asked of.** -/
def noAddress : Address := ⟨fun _ => 0, ⟨1, 0⟩, { key := .reg "", schema := "", pos := 0 }⟩

def addressAt (i : ℕ) : Address :=
  match bitAtGate h i, bitAtGate h (wireMap h i) with
  | some (t, r), some (tw, rq) => ⟨fun _ => portNamed h t r.row, placeAt h tw rq, slotAt h tw rq⟩
  | _, _ => noAddress

open Classical in
noncomputable def addressOf (i : ℕ) : Address :=
  if ∀ j, wireMap h (wireMap h j) = wireMap h j then addressAt h i else noAddress

theorem addressOf_eq (hω : ∀ j, wireMap h (wireMap h j) = wireMap h j) (i : ℕ) :
    addressOf h i = addressAt h i := by
  classical
  exact if_pos hω

theorem addressOf_shares (i : ℕ) :
    (addressOf h i).place = (addressOf h (wireMap h i)).place ∧
      (addressOf h i).slot = (addressOf h (wireMap h i)).slot := by
  classical
  by_cases hω : ∀ j, wireMap h (wireMap h j) = wireMap h j
  · rw [addressOf_eq h hω, addressOf_eq h hω]
    unfold addressAt
    rw [hω i]
    cases hi : bitAtGate h i with
    | none =>
      have hs : (sourceBits h.ms).find? (·.1 == i) = none := by
        unfold bitAtGate at hi
        cases hf : (sourceBits h.ms).find? (·.1 == i) with
        | none => rfl
        | some s => simp [hf] at hi
      rw [wireMap_of_not_source h hs, hi]
      exact ⟨rfl, rfl⟩
    | some a =>
      obtain ⟨t, r⟩ := a
      cases hg : bitAtGate h (wireMap h i) with
      | none => exact ⟨rfl, rfl⟩
      | some b =>
        obtain ⟨tw, rq⟩ := b
        exact ⟨rfl, rfl⟩
  · have e : ∀ j, addressOf h j = noAddress := fun j => if_neg hω
    rw [e i, e (wireMap h i)]
    exact ⟨rfl, rfl⟩

end SpotFlock
