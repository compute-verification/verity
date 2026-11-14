module

public import Proofs.Flock.SpotFlock.HeldOfFiles

/-!
# What the two-stage form would decide of the held copies

`HeldOf.heldOf` is noncomputable (its frame is SPOT's own, built from a code), so a check can't call it. `heldOk i it u`
decides whether it builds a held copy, at the copies before SPOT's frame is set (`held₀`): `heldOf i it u` is `some`
exactly where `heldOk i it u` holds (`heldOf_isSome`), since nothing `decided` reads is the frame (`decided_withFrame`).
A form that refuses unless `heldOk` holds at the run of record's input and interior and the members' reference
statements gets a held copy at every run it accepts, and at an accepted run `TwoHolds`
(`HeldOfRun.twoHolds_of_files`).

## What deciding it costs

`decided` reads the descriptor's checks (`descCore`): the rule and count checks, the layout and the first stage's design
(`Slots.layout`, `slotDesign`), a step per stratum (`designOk`), and per template its reference statement's reading
(`templateOk`: parsing its circuit file, the session verifier's own parse) and its member's size class (`memberOk`, a
closed form in the unit's descriptor); and `two`'s `membersScoped`, a step per template, and `wiringOk`, each clause
of which reads the circuit files and the registered wiring once per template, whatever the count of units.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs
open Lean (Json)

/-! ## Nothing `decided` reads is the frame -/

theorem two_setFrame (h : Held) (F : Code) (s : AuditStatement) : two { h with frame := F } s = two h s := rfl

theorem descCore_setFrame (h : Held) (F : Code) : descCore { h with frame := F } = descCore h := rfl

theorem decided_setFrame (u : Unread) (it : Spot.Run.Interior) (h : Held) (F : Code) :
    decided u it { h with frame := F } = decided u it h := rfl

theorem two_withFrame (h : Held) (c₀ : Code) (s : AuditStatement) : two (withFrame h c₀) s = two h s :=
  two_setFrame h _ s

theorem descCore_withFrame (h : Held) (c₀ : Code) : descCore (withFrame h c₀) = descCore h :=
  descCore_setFrame h _

theorem decided_withFrame (u : Unread) (it : Spot.Run.Interior) (h : Held) (c₀ : Code) :
    decided u it (withFrame h c₀) = decided u it h :=
  decided_setFrame u it h _

/-! ## The check -/

/-- **Whether `heldOf` builds a held copy**, decided: the boundary registration, the cells and the layout read, the
reference statements are in Flock's masked scope and `decided` holds at the copies before SPOT's frame is set. -/
def heldOk (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) : Bool :=
  match firstOf i, cellsOf i it with
  | some first, some c =>
    match (Spot.Slots.layout first.program first.query).toOption with
    | some l =>
      if hz : xsOk u.xs = true then decided u it (held₀ i it u first c.1 c.2 (requests l) (zkOk_refStmt hz))
      else false
    | none => false
  | _, _ => false

theorem isSome_keep (u : Unread) (it : Spot.Run.Interior) (h : Held) : (keep u it h).isSome = decided u it h := by
  unfold keep
  split <;> simp_all

theorem heldOf_isSome (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) :
    (heldOf i it u).isSome = heldOk i it u := by
  unfold heldOf heldOk
  cases firstOf i with
  | none => rfl
  | some first =>
    cases cellsOf i it with
    | none => rfl
    | some c =>
      dsimp only [Option.bind]
      cases (Spot.Slots.layout first.program first.query).toOption with
      | none => rfl
      | some l =>
        dsimp only [Option.bind]
        by_cases hz : xsOk u.xs = true
        · rw [dif_pos hz, dif_pos hz, isSome_keep, decided_withFrame]
        · rw [dif_neg hz, dif_neg hz]
          rfl

theorem heldOf_of_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} (u : Unread) (hk : heldOk i it u = true) :
    ∃ h, heldOf i it u = some h := by
  rw [← heldOf_isSome] at hk
  exact Option.isSome_iff_exists.1 hk

/-- **The form's check of the held copies**: `heldOk` at the members' reference statements on the blobs
(`refStmtsOf`), with the registered copies `u` gives of the rest (its reference statements are not read); false where
one is missing. At an accepted run it gives `TwoHolds` (`HeldOfRun.twoHolds_of_files`). -/
def heldOkDesc (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input)
    (it : Spot.Run.Interior) (u : Unread) : Bool :=
  match refStmtsOf p bs with
  | some xs => heldOk i it { u with xs }
  | none => false

end SpotFlock.HeldOf
