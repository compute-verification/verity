module

public import Proofs.Flock.SpotFlock.HeldOfFiles

/-!
# What the two-stage form would decide of the held copies

`HeldOf.heldOf` is noncomputable (its frame is SPOT's own, built from a code), so a check can't call it. `heldOk i it xs`
decides, computably, whether it builds a held copy: `heldOf i it u` is `some` exactly where `heldOk i it u.xs` holds
(`heldOf_isSome`), whatever `u`'s descriptor, query, code and rounds. A form that refuses unless `heldOk` holds at the
run of record's input and interior and the members' reference statements gets a held copy at every run it accepts,
and with the run's population check `TwoHolds` (`HeldOfRun.twoHolds_of_files`).

## What deciding it costs

`heldOk` decides the descriptor's checks at the built copies (`descCoreAt`): the rule and count checks, the layout and
the first stage's design (`Slots.layout`, `slotDesign`), a step per stratum (`designOk`), and per template its
reference statement's reading (`templateOk`: parsing its circuit file, the session verifier's own parse) and its
member's size class (`memberOk`, a closed form in the unit's descriptor). None takes a step per unit or per gate.
-/

@[expose] public section

namespace SpotFlock.HeldOf

open Claims Claims.Sampling SamplingDeploy Properties.Spot SpotFlock.RunsAs
open Lean (Json)

/-- `templateOk` at the reference statement `xS`. -/
def templateOkAt (xS : Msg) : Bool :=
  match FlockDeploy.readOf true xS with
  | some r =>
    r.tags.hiddenOutputs && !r.tags.typed &&
      (match _root_.Flock.Desc.parseStmt r.tags r.I.circuitFile r.I.tables with
        | .ok _ => true
        | .error _ => false) &&
      decide (r.stmt.unit.unitLog ≤ 32) && (shapeOf xS).length == 1 &&
      r.stmt.ports.all _root_.Flock.ProvedScope.v2Row
  | none => false

/-- `descCore` at the registration `g`, the members `ms`, the second stage's counts `k` and the reference statements
`xS`. -/
def descCoreAt (g : RegTwo) (ms : Spot.Bits.Members) (k : List ℕ) (xS : ℕ → Msg) : Bool :=
  match (Spot.Slots.layout g.first.program g.first.query).toOption,
      (slotDesign g.first.rule g.first.program g.first.query g.first.table).toOption with
  | some l, some d => ms.all memberOk && designOk d l && k.length == ms.length && k.all (1 ≤ ·) &&
      g.ns.size == ms.length && (List.range ms.length).all (fun t => templateOkAt (xS t))
  | _, _ => false

theorem descCore_eq (h : Held) : descCore h = descCoreAt h.reg h.ms h.k h.xS₀ := rfl

/-- `two` at the registration `g`, the members `ms`, the second stage's counts `k` and the reference statements
`xS`. -/
def twoAt (g : RegTwo) (ms : Spot.Bits.Members) (k : List ℕ) (xS : ℕ → Msg) : Option Design.Two :=
  match (Spot.Slots.layout g.first.program g.first.query).toOption,
      (slotDesign g.first.rule g.first.program g.first.query g.first.table).toOption with
  | some l, some d =>
    if scopeOk (bitsProgs ms) (membersPart ms) ⟨filesReplay ms g.ns l, d, k⟩ &&
        filesOk g ms l k && (List.range ms.length).all (fun t => templateOkAt (xS t)) then
      some ⟨filesReplay ms g.ns l, d, k⟩
    else none
  | _, _ => none

theorem two_eq (h : Held) (s : AuditStatement) : two h s = twoAt h.reg h.ms h.k h.xS₀ := rfl

/-- **What `decided` reads of the copies `heldOf` builds**, at the boundary registration `first`, the cells `c` and the
layout `l`. -/
def decidedAt (it : Spot.Run.Interior) (xs : List Msg) (first : Reg) (c : Array ℕ × Array ℕ)
    (l : Spot.Slots.SlotOrder) : Bool :=
  xs.length == it.circuits.size &&
    decide ((List.range (msOf xs (unitsOf (requests l) c.1 it.circuits.size)).length).map fun t =>
      (it.circuits.map (·.1)).toList.getD t "").Nodup &&
    (namesOf it.params.verifier).values.isEmpty &&
    descCoreAt ({ first, ns := c.1, ks := c.2,
             blockOf := filesBlockOf (msOf xs (unitsOf (requests l) c.1 it.circuits.size)) c.1,
             always := fun _ => false } : RegTwo)
      (msOf xs (unitsOf (requests l) c.1 it.circuits.size)) c.2.toList (refStmt xs)

/-- **Whether `heldOf` builds a held copy**, decided: the boundary registration, the cells and the layout read, the
reference statements are in Flock's masked scope and `decidedAt` holds. -/
def heldOk (i : Spot.Run.Input) (it : Spot.Run.Interior) (xs : List Msg) : Bool :=
  match firstOf i, cellsOf i it with
  | some first, some c =>
    match (Spot.Slots.layout first.program first.query).toOption with
    | some l => xsOk xs && decidedAt it xs first c l
    | none => false
  | _, _ => false

theorem isSome_keep (u : Unread) (it : Spot.Run.Interior) (h : Held) : (keep u it h).isSome = decided u it h := by
  unfold keep
  split <;> simp_all

theorem decided_held₀ (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) (first : Reg)
    (c : Array ℕ × Array ℕ) (l : Spot.Slots.SlotOrder) (hz : ∀ t, zkOk (refStmt u.xs t) = true) :
    decided u it (withFrame (held₀ i it u first c.1 c.2 (requests l) hz) u.c₀) = decidedAt it u.xs first c l := rfl

theorem heldOf_isSome (i : Spot.Run.Input) (it : Spot.Run.Interior) (u : Unread) :
    (heldOf i it u).isSome = heldOk i it u.xs := by
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
        · rw [dif_pos hz, isSome_keep, decided_held₀, hz, Bool.true_and]
        · rw [dif_neg hz]
          simp [hz]

theorem heldOf_of_ok {i : Spot.Run.Input} {it : Spot.Run.Interior} (u : Unread) (hk : heldOk i it u.xs = true) :
    ∃ h, heldOf i it u = some h := by
  rw [← heldOf_isSome] at hk
  exact Option.isSome_iff_exists.1 hk

/-- **The form's check of the held copies**: `heldOk` at the members' reference statements on the blobs
(`refStmtsOf`); false where one is missing. It takes a step per template and per stratum, none per unit or gate, and
with the run's population check gives `TwoHolds` (`HeldOfRun.twoHolds_of_files`). -/
def heldOkDesc (p : VStar.Recursive.SampledTwo.Params) (bs : Core.Blobs) (i : Spot.Run.Input)
    (it : Spot.Run.Interior) : Bool :=
  match refStmtsOf p bs with
  | some xs => heldOk i it xs
  | none => false

end SpotFlock.HeldOf
