/-
# The algebraic phase's scheduled rounds (live PROTOCOL.md §10.1), a function of the statement

A firewalled rep's algebraic phase (the masked zerocheck and lincheck) has a schedule the statement fixes: which frames
each round holds, in what order, and how many coins it squeezes. Each observed value is masked downstream, so the round an
honest device frames is the schedule with every observed value zero, and a failed rep releases those rounds in the
device's place. This is that schedule:

* `rounds`: `Finish.repPiop`'s rounds at zero values (the PIOP `Finish.openLive` runs), on the statement's setup, its
  `--zk` schedule's `k0`, level 0's cap and the rep's domain, but for the pads' label and root, which the masking
  challenger inserts itself (`padsFrames`). The coins it computes them on are `coin`'s; no round's frames depend on them.
-/
import Flock.Firewall.Finish

namespace Flock.Firewall.Skeleton

open Flock Transcript Finish

/-- Coin `i` of the phase, counted from `1` over every coin it draws: distinct, neither `0` nor `1` (the circuit fold
divides by `1 + r`), and Rust's `Recorder`'s, so both sides fold the circuit on the same coins. -/
def coin (i : Nat) : F128 := ⟨0x5eed0000 + i.toUInt64, 0x123456789abcdef0⟩

/-- The rounds recorded so far, each its framed bytes and coin count, and the count of coins drawn. -/
abbrev RecM := StateT (Array (ByteArray × Nat) × Nat) V

/-- **The recording `Asks`**: each round kept with its coin count (its squeeze header's last eight bytes), answered by
the next `coin`s. -/
def recAsks : Asks RecM where
  link _ _ := throw "skeleton: a link message in the algebraic phase"
  round c := do
    let (rs, k) ← get
    let n := (u64OfLE c (c.size - 8)).toNat
    set (rs.push (c, n), k + n)
    return fun i => coin (k + i + 1)

/-- The pads' label and their root `root`, as the masking challenger inserts them after the zerocheck's label
(`Frame.Masker.label`): a device's round frames neither. -/
def padsFrames (root : ByteArray) : ByteArray := Frame.labelFrame Zk.LABEL_PADS.toUTF8 ++ Frame.bytesFrame root

/-- What the first round frames before the pads: the domain `dom`, the statement's label, digest and cap, and the
zerocheck's label (`Finish.zerocheckFw`'s order). -/
def head (st : Setup) (cap dom : ByteArray) : ByteArray :=
  (Transcript.init #[] dom).pending ++ Frame.labelFrame "flock-r1cs-v0".toUTF8 ++ Frame.bytesFrame st.digest ++
    Frame.bytesFrame cap ++ Frame.labelFrame Frame.LABEL_ZC.toUTF8

/-- **The algebraic phase's scheduled rounds** of a rep whose transcript starts at `dom`, on the statement `st` at its
`--zk` schedule's `k0` and level 0's cap `cap`: each round as an honest device frames it, every observed value zero, with
its coin count. They are `repPiop`'s rounds at zero values, the first without the pads' frames (`padsFrames`). -/
def rounds (st : Setup) (k0 : Nat) (cap dom : ByteArray) : V (Array (ByteArray × Nat)) := do
  let lay := Zk.PadLayout.new st.m st.kLog k0
  let root := ByteArray.mk (Array.replicate 64 0)
  let prog : TMW RecM Unit := do
    let _ ← repPiop recAsks st lay cap root #[] fun _ => 0
  let (_, rs, _) ← (prog.run' (Transcript.init #[] dom)).run (#[], 0)
  let some (r0, n0) := rs[0]? | throw "skeleton: the algebraic phase has no round"
  let pre := head st cap dom
  let pads := padsFrames root
  if (r0.extract 0 pre.size).data != pre.data ∨ (r0.extract pre.size (pre.size + pads.size)).data != pads.data then
    throw "skeleton: the first round does not open with the statement, the zerocheck's label and the pads"
  return rs.set! 0 (pre ++ r0.extract (pre.size + pads.size) r0.size, n0)

end Flock.Firewall.Skeleton
