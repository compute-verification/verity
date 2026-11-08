/-
`Frames.askOf`, the answer to a prover's `place`, against its rep's own requests (`Frames.roundStep`,
`Frames.coinsStep`), on a rep of two masked values in two scheduled one-value rounds: a place read before the rep's
round and its coins answers from what it read; the frame after them is theirs alone, so the next round leaves under the
next pad. That frame put back as the place read it (with the place made, `placeOf`'s frame) rolls the masker back: the
next round leaves under the pad the last one used, and the two releases' sum is the two device values' sum. `askOf`
writes no frame, so nothing but a rep's round and coins puts its frame back.
-/
module

public import Flock.Firewall.Frames
meta import Flock.Firewall.Frames

@[expose] public section

namespace Flock.Test.FramesAsk

open Flock Flock.Firewall Flock.Firewall.Frames
open Flock.Transcript (header)

def h0 : F128 := ⟨0x1111, 0x2222⟩
def h1 : F128 := ⟨0x3333, 0x4444⟩

/-- Two pads, absorbed in order, both in the algebraic phase. -/
def cfg : Frame.Cfg := { (default : Frame.Cfg) with h := #[h0, h1], order := #[0, 1], nPiop := 2 }

/-- A round observing the one value `v` and squeezing one coin. -/
def roundOf (v : F128) : ByteArray := F128.pushBytes (header 3 1 1) v ++ header 4 1 1

def sched : ByteArray := roundOf 0

def v : F128 := ⟨5, 6⟩
def v' : F128 := ⟨7, 9⟩

/-- Table 0's rep 0 inside the masked phase, its two scheduled rounds computed. -/
def rep₀ : Rep :=
  { sid := 0, table := 0, rep := 0, cfg, st := { m := { on := true }, begun := true },
    alg := #[(sched, 1), (sched, 1)] }

def spec : SessionSpec := { (default : SessionSpec) with tables := #["t"] }

def R₀ : Sessions := { sessions := #[{ sid := 0, spec, sts := #[default] }], reps := #[rep₀] }

def L₀ : Level0.Slots :=
  { slots := #[{ l0 := default, owner := 0, table := 0, bound := some (0, "t") }], sessions := #[("s", 0)],
    tables := #[(0, 1)] }

/-- The rep's device round of the value `x` in `R`: what leaves, and the frames with the rep's frame put back. -/
def roundAt (R : Sessions) (x : F128) : Except String ByteArray × Sessions :=
  let o := roundStep R L₀ 0 0 0 .empty (roundOf x) 1
  (o.out, match o.put with | some p => R.put p | none => R)

/-- The rep's coins for its last round in `R`, its frame put back. -/
def coinsAt (R : Sessions) : Sessions :=
  match coinsStep R 0 0 0 #[⟨7, 0⟩] with
  | .ok p => R.put p
  | .error _ => R

def asked := askOf R₀ L₀ "place s 0 0"
def R₁ := (roundAt R₀ v).2
def R₂ := coinsAt R₁

/-- The rep's frame as a place read it in `R₀`, the place made. -/
def stale : Option Rep := do
  let (_, p) ← R₀.find 0 0 0
  let s ← R₀.sessions[0]?
  let l0 ← Level0.ofSession L₀ 0 "t"
  let p ← (withAlg s l0 "t" p).toOption
  let (_, _, p') ← (placeOf R₀.salts l0 s p (s.spec.domain "t" 0).toUTF8).toOption
  return p'

/-- `R₂` with that frame put back. -/
def R₃ := match stale with | some p => R₂.put p | none => R₂

def wordOf : Except String ByteArray → Option F128
  | .ok b => (Frame.words b 16 1)[0]?
  | .error _ => none

def posOf (R : Sessions) : Option (Nat × Nat) := (R.find 0 0 0).map fun (_, p) => (p.coins.size, p.st.m.pos)

#guard (asked.toOption.map (·.1)) == some "alg 1"
#guard wordOf (roundAt R₀ v).1 == some (v + h0)
#guard posOf R₂ == some (1, 1)
#guard (askOf R₂ L₀ "place s 0 0").toOption.map (·.1) == some "alg 1"
#guard wordOf (roundAt R₂ v').1 == some (v' + h1)
#guard stale.isSome
#guard posOf R₃ == some (0, 0)
#guard wordOf (roundAt R₃ v').1 == some (v' + h0)
#guard (wordOf (roundAt R₃ v').1).map (· + (v + h0)) == some (v + v')

end Flock.Test.FramesAsk
