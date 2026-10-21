/-
# The session loop's frames: which rounds leave, and the coins each frame takes

The session loop (`Session`) serves the reps' frames (`Frame.serveWith`) on connections of its own, one slot each, and
holds a session's rounds to them. A slot keeps the round its frame last released (`released`) until a session's `Round`
carries it, the session and stream its rounds go to once one did, and the coins the loop opened for the round it carried:

* `carry`: a session's `Round` of a rep stream leaves only as the release of the frame its stream is bound to, byte for
  byte and at its coin count; at the stream's first `Round`, where no frame is bound to it, as the stream's domain frame
  (`domainFrame`, `Transcript.init`'s) followed by the release of a frame bound to none, which is then bound to it.
  Anything else no Lean frame released, and the loop refuses it.
* `feed`: once the coins answered that round open the committed tree (`State.answer`), the loop sets the slot's coins to
  `feedOf` the answer before the prover receives it, so they are set before the frame is asked to take them.
* `take`: the frame's `coins N` takes the slot's coins, and only them: `N` of them, the bytes the request carries equal
  to theirs. With none set, or others carried, the frame refuses.

So the coins a frame takes are the loop's own reading of the auditor's answer (`feedOf`), never the prover's.
-/
import Flock.Firewall.Frame

namespace Flock.Firewall.Frames

open Frame (words wordBytes)

/-- The most coins of one round a frame is fed. -/
def COIN_MAX : Nat := 65536

/-- **The coins the loop feeds a frame** for an opening `a` of `n` coins (`Resp::Coins`' frame): its first `n` words,
16 bytes each after the tag and count, as `Coins.check` cuts them, up to `COIN_MAX`. -/
def feedOf (n : Nat) (a : List UInt8) : Array F128 := words ⟨(a.drop 5).toArray⟩ 0 (min n COIN_MAX)

/-- A stream's domain frame, which its first round begins with (`Transcript.init`, `frame_domain`). -/
def domainFrame (d : String) : ByteArray :=
  let b := d.toUTF8
  Transcript.header 1 0 b.size ++ Transcript.padded b

/-- A frame's slot. -/
structure Slot where
  /-- the round the frame last released and its coin count, until a session's `Round` carries it -/
  pending : Option (ByteArray × Nat) := none
  /-- the session and stream the frame's rounds go to, once one did -/
  bound : Option (Nat × String) := none
  /-- the coins the loop opened for the round last carried, until the frame takes them -/
  fed : Option (Array F128) := none
  closed : Bool := false
  deriving Inhabited

abbrev Registry := Array Slot

/-- The frame of slot `i` released `out`, which squeezes `n` coins. -/
def release (R : Registry) (i : Nat) (out : ByteArray) (n : Nat) : Registry :=
  R.modify i fun s => { s with pending := some (out, n), fed := none }

/-- **Whether session `sid`'s `Round` of `stream`, `content` squeezing `n` coins, leaves**, the stream's domain frame
being `dom`: the slot it goes to and the registry after, or why it does not. -/
def carry (R : Registry) (sid : Nat) (stream : String) (dom content : ByteArray) (n : Nat) :
    Except String (Registry × Nat) :=
  match R.findIdx? (·.bound == some (sid, stream)) with
  | some i =>
    let s := R[i]!
    if s.closed then .error s!"a round of {stream}, whose Lean frame ended"
    else match s.pending with
      | some (out, k) =>
        if out == content && k == n then .ok (R.set! i { s with pending := none, fed := none }, i)
        else .error s!"a round of {stream} that is not its Lean frame's release"
      | none => .error s!"a round of {stream} its Lean frame did not release"
  | none =>
    let first := fun (s : Slot) => s.bound.isNone && !s.closed &&
      s.pending.any fun (out, k) => k == n && dom ++ out == content
    match R.findIdx? first with
    | some i => .ok (R.set! i { R[i]! with pending := none, fed := none, bound := some (sid, stream) }, i)
    | none => .error s!"a round of {stream} no Lean frame released"

/-- The coins `c` the loop opened for the round slot `i` carried. -/
def feed (R : Registry) (i : Nat) (c : Array F128) : Registry :=
  R.modify i fun s => { s with fed := some c }

/-- **The coins slot `i`'s frame takes** for `coins k` followed by the bytes `b`: the slot's, which `b` must be. -/
def take (R : Registry) (i k : Nat) (b : ByteArray) : Except String (Registry × Array F128) :=
  match R[i]? with
  | none => .error "no such frame"
  | some s => match s.fed with
    | none => .error "coins for a round whose coins the session loop has not opened"
    | some c =>
      if c.size != k then .error s!"{k} coins where the session loop opened {c.size}"
      else if wordBytes c != b then .error "coins that are not the ones the session loop opened"
      else .ok (R.set! i { s with fed := none }, c)

/-- Slot `i`'s frame ended: no round of it leaves after. -/
def close (R : Registry) (i : Nat) : Registry :=
  R.modify i fun s => { s with closed := true, pending := none }

/-- A new slot, and its index. -/
def openSlot (R : Registry) : Registry × Nat := (R.push {}, R.size)

/-- The hooks a frame in slot `i` of the registry `reg` runs beside (`Frame.Hooks`). -/
def hooks (reg : IO.Ref Registry) (i : Nat) : Frame.Hooks where
  released out n := reg.modify (release · i out n)
  coins k b := reg.modifyGet fun R => match take R i k b with
    | .ok (R', c) => (.ok c, R')
    | .error e => (.error e, R)

end Flock.Firewall.Frames
