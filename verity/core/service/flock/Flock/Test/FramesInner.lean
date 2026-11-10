/-
A rep's inner rounds in the loop's frames (`Frames.roundPlace`, `Frames.roundStep`), at table 0's rep 0 past its
opening: the inner proof's round `j` has a place, the frame's own round on the loop's finish's (`Frame.innerSched`),
only once the finish computed it (`Sessions.innerRound`). A host that relays round `j` before then, whatever it relays,
is refused for it, and the rep's frame is not put back, so the session stops on the refusal (`Session`'s `Round`).
-/
module

public import Flock.Firewall.Frames
meta import Flock.Firewall.Frames

@[expose] public section

namespace Flock.Test.FramesInner

open Flock Flock.Firewall Flock.Firewall.Frames

def spec : SessionSpec := { (default : SessionSpec) with tables := #["t"] }

/-- Table 0's rep 0 past its opening, at its inner proof's round `j`. -/
def repAt (j : Nat) : Rep :=
  { sid := 0, table := 0, rep := 0, cfg := default, st := { m := { on := true, inner := some j }, begun := true },
    opened := true }

/-- The loop's frames with the rep at its inner round `j`, the loop's finish having computed `k` of its inner rounds. -/
def R (j k : Nat) : Sessions :=
  { sessions := #[{ sid := 0, spec, sts := #[default] }], reps := #[repAt j],
    inner := #[(.empty, 0, Array.replicate k .empty)] }

def L₀ : Level0.Slots := { slots := #[{ l0 := default, owner := 0, table := 0, bound := some (0, "t") }] }

/-- What the host relays as the round: anything. -/
def relay : ByteArray := ByteArray.mk #[1, 2, 3]

/-- Why the frame takes no device round in the round's place, if it takes none; or why the round has no place. -/
def placed (j k : Nat) : Except String (Option String) :=
  (roundPlace (R j k) L₀ 0 0 0 .empty relay 1).map (·.2.2.2)

def step (j k : Nat) : RoundStep := roundStep (R j k) L₀ 0 0 0 .empty relay 1

def errOf {α : Type} : Except String α → Option String
  | .error e => some e
  | .ok _ => none

def why (j : Nat) : String := s!"table 0 rep 0's inner round {j}, which the loop's finish has not computed"

#guard errOf (placed 0 0) == some (why 0)
#guard errOf (placed 1 1) == some (why 1)
#guard errOf (placed 3 2) == some (why 3)
#guard errOf (step 1 1).out == some (why 1) && (step 1 1).put.isNone
#guard placed 0 1 matches .ok none
#guard placed 1 2 matches .ok none
#guard placed 3 4 matches .ok none

end Flock.Test.FramesInner
