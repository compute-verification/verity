/-
A session's stop in the loop's frames (`Frames.roundStep`, `Frames.coinsStep`, `Frames.finishOf`), on a table's two
reps of one masked value each, in one scheduled one-value round: rep 0's device round fails (the place's squeeze alone,
as a failed worker's), so the scheduled round leaves in its place, its value as rep 0's pad alone. Rep 1's next device
round would pass against its own scheduled round, and does in a session with no stop, leaving as its value under rep 1's
pad; after rep 0's failure it leaves as the scheduled round of a stopped rep, byte for byte `Frame.standIn`'s (the
model's release of every round after a session's first failure), its value as rep 1's pad alone. Both reps' frames
then answer `finish` `stop`, so the session's `Finish` leaves without proofs.
-/
module

public import Flock.Firewall.Frames
meta import Flock.Firewall.Frames

@[expose] public section

namespace Flock.Test.FramesStop

open Flock Flock.Firewall Flock.Firewall.Frames
open Flock.Transcript (header)

def h0 : F128 := ⟨0x1111, 0x2222⟩
def h1 : F128 := ⟨0x3333, 0x4444⟩

/-- One pad `h`, absorbed in the algebraic phase. -/
def cfgOf (h : F128) : Frame.Cfg := { (default : Frame.Cfg) with h := #[h], order := #[0], nPiop := 1 }

/-- A round observing the one value `v` and squeezing one coin. -/
def roundOf (v : F128) : ByteArray := F128.pushBytes (header 3 1 1) v ++ header 4 1 1

def sched : ByteArray := roundOf 0

/-- A failed worker's round: the place's squeeze alone. -/
def squeeze : ByteArray := header 4 1 1

def v : F128 := ⟨5, 6⟩

def m₀ : Frame.Masker := { on := true }

/-- Table 0's rep `r` inside the masked phase, its scheduled round computed. -/
def repOf (r : Nat) (h : F128) : Rep :=
  { sid := 0, table := 0, rep := r, cfg := cfgOf h, st := { m := m₀, begun := true }, alg := #[(sched, 1)] }

def spec : SessionSpec := { (default : SessionSpec) with tables := #["t"] }

def R₀ : Sessions := { sessions := #[{ sid := 0, spec, sts := #[default] }], reps := #[repOf 0 h0, repOf 1 h1] }

def L₀ : Level0.Slots :=
  { slots := #[{ l0 := default, owner := 0, table := 0, bound := some (0, "t") }], sessions := #[("s", 0)],
    tables := #[(0, 1)] }

/-- Rep `r`'s device round `x` in `R`: what leaves, and the frames with the rep's frame put back. -/
def roundAt (R : Sessions) (r : Nat) (x : ByteArray) : Except String ByteArray × Sessions :=
  let o := roundStep R L₀ 0 0 r .empty x 1
  (o.out, match o.put with | some p => R.put p | none => R)

/-- Rep `r`'s coins for its last round in `R`, its frame put back. -/
def coinsAt (R : Sessions) (r : Nat) : Option Sessions :=
  (coinsStep R 0 0 r #[⟨7, 0⟩]).toOption.map R.put

def wordOf : Except String ByteArray → Option F128
  | .ok b => (Frame.words b 16 1)[0]?
  | .error _ => none

def stoppedOf (R : Sessions) (r : Nat) : Option Bool := (R.find 0 0 r).map (·.2.st.m.stopped)

/-- `R` with every rep's frame at its stream's end (its masked phase closed, its inner proof's rounds all left), each
`stopped` as it is. -/
def ended (R : Sessions) : Sessions :=
  { R with reps := R.reps.map fun p =>
      { p with st := { p.st with m := { p.st.m with done := true, inner := some Frame.INNER_ROUNDS } } } }

def finishStop : Except String Unit → Bool
  | .error e => e.startsWith Frame.STOP
  | .ok () => false

/-- Whether rep `r`'s frame in `R` answers `finish` `stop`. -/
def repStops (R : Sessions) (r : Nat) : Bool :=
  match R.find 0 0 r with
  | some (_, p) => ((Frame.step p.cfg p.st .finish).1 matches .done true)
  | none => false

def R₁ := (roundAt R₀ 0 squeeze).2
def R₂ := coinsAt R₁ 0
def out₁ := (roundAt (R₂.getD R₀) 1 (roundOf v)).1
def R₃ := coinsAt (roundAt (R₂.getD R₀) 1 (roundOf v)).2 1

/-- The session with no stop: both reps' device rounds pass, and their coins come. -/
def H := (coinsAt (roundAt R₀ 0 (roundOf v)).2 0).bind fun R => coinsAt (roundAt R 1 (roundOf v)).2 1

/-- The model's release of rep 1's round in a stopped session: the scheduled round at a stopped rep's state. -/
def standIn : Option ByteArray := (Frame.standIn (cfgOf h1) m₀ sched 1 "").toOption.map (·.1)

#guard wordOf (roundAt R₀ 0 squeeze).1 == some h0
#guard stoppedOf R₁ 0 == some true && R₁.stopped 0
#guard R₂.isSome
#guard !R₀.stopped 0
#guard wordOf (roundAt R₀ 1 (roundOf v)).1 == some (v + h1)
#guard standIn.isSome
#guard out₁.toOption == standIn
#guard wordOf out₁ == some h1
#guard R₃.bind (stoppedOf · 1) == some true
#guard R₃.map (fun R => repStops (ended R) 0 && repStops (ended R) 1) == some true
#guard R₃.map (fun R => finishStop (finishOf (ended R) 0 1)) == some true
#guard H.map (fun R => !R.stopped 0 && (finishOf (ended R) 0 1 matches .ok ())) == some true

end Flock.Test.FramesStop
