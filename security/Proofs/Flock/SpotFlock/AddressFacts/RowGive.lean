module

public import Proofs.Flock.SpotFlock.AddressFacts.Row
public import Proofs.Flock.SpotFlock.AddressFacts

/-! What gives the row fact of the address premise (`RowGoal`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable (h : Held) (ω : ℕ → ℕ) (keep : ℕ → Bool)

def refPortsNamed : Bool :=
  (writtenBits h.ms).all fun s =>
    (match refEntry h s.2.1 s.2.2.row with
      | .null => true
      | _ => false) ||
      (heldLink h (h.names.getD s.2.1 "") s.2.2.row).isSome ||
      (heldWire h (h.names.getD s.2.1 "") s.2.2.row).isSome

def ownShape (u : ℕ) : RowShape :=
  ⟨cellPos h u, h.reg.ns.getD (templateOf h.ms u) 0, 0⟩

def linkShape (u : ℕ) (port : String) : RowShape := rowShape h (fun _ _ _ => 0) u port .boundaryLink

def wireShape (u : ℕ) (port : String) : RowShape := rowShape h (fun _ _ _ => 0) u port .wired

def RowShape.FitsSized (sh : RowShape) (s : Slot) : Prop :=
  s.pos = sh.pos ∧ s.stride = sh.stride ∧ s.units = sh.units ∧ s.sized = true

theorem leavesAt_fits {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.Fits s) :
    leavesAt roots s e = e.leaves := by
  unfold leavesAt; rw [hf.2.2.2]; rfl

theorem leavesAt_sized {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.FitsSized s) :
    leavesAt roots s e = unitsAt roots s := by
  unfold leavesAt; rw [hf.2.2.2]; rfl

theorem leafAt_sized {roots : List Msg} {s : Slot} {sh : RowShape} (e : Entry) (hf : sh.FitsSized s)
    (hu : 0 < unitsAt roots s) : leafAt roots s e = s.pos + placeOf roots * s.stride := by
  unfold leafAt
  split
  · omega
  · rw [leavesAt_sized e hf, Nat.div_self hu, Nat.mul_one]

variable (addr : ℕ → Address)

def NeededKinds : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → ∀ reg,
    regNamed r.ins r.sts ((addr i).port r.u) = some reg →
      kindHeld h (templateOf h.ms r.u) reg.name reg.value = .boundaryLink ∨
        kindHeld h (templateOf h.ms r.u) reg.name reg.value = .wired

def ShapeFact : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    (∀ reg, regNamed r.ins r.sts ((addr i).port r.u) = some reg →
      (kindHeld h (templateOf h.ms r.u) reg.name reg.value = .boundaryLink →
          (linkShape h r.u reg.name).Fits (addr i).slot) ∧
        (kindHeld h (templateOf h.ms r.u) reg.name reg.value = .wired →
          (wireShape h r.u reg.name).FitsSized (addr i).slot)) ∧
    (regNamed r.ins r.sts ((addr i).port r.u) = none → (ownShape h r.u).FitsSized (addr i).slot)

def OutputsAt : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i →
    regNamed r.ins r.sts ((addr i).port r.u) = none → _root_.Flock.HmOut.nIn r.sts[0]!.c ≤ (addr i).port r.u

def RowGoal : Prop :=
  (∀ v, h.reg.always v = false) → NeededKinds h addr → ShapeFact h addr → OutputsAt h addr →
    (∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → r.RootsHold h ω keep (addr i) (ω i)) →
    SpotFlock.RowFact h ω keep addr

end AddressRow

end SpotFlock
