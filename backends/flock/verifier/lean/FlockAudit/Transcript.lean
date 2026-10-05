import FlockAudit.Frame
import FlockAudit.Prog

/-!
# A run's transcript as bytes

One record per event, in order: a tag byte, then frames. Tag `1` is a reply as it came (the channel, the request, the
reply), and tag `2` a draw (its bytes). The reader refuses bytes that are not a whole transcript.
-/

namespace FlockAudit.Transcript

open Prog

def encodeEvent : Event → ByteArray
  | .asked chan body reply => ByteArray.mk #[1] ++ Frame.encode chan.toUTF8 ++ Frame.encode body ++ Frame.encode reply
  | .drew b => ByteArray.mk #[2] ++ Frame.encode b

def write (path : System.FilePath) (es : Array Event) : IO Unit :=
  IO.FS.withFile path .write fun h => es.forM fun e => h.write (encodeEvent e)

private def frame (b : ByteArray) (off : Nat) : Except String (ByteArray × Nat) :=
  match Frame.take b off with
  | some x => .ok x
  | none => .error s!"transcript: a frame at byte {off} runs past the end"

/-- The events of a transcript. -/
def decode (b : ByteArray) : Except String (List Event) :=
  -- each record takes at least one byte, so `b.size + 1` records' fuel reads any transcript
  let rec go : Nat → Nat → Array Event → Except String (List Event)
    | 0, _, _ => throw "transcript: more records than bytes"
    | fuel + 1, off, acc => do
      if off == b.size then return acc.toList
      match b.get! off with
      | 1 =>
        let (c, o1) ← frame b (off + 1)
        let (body, o2) ← frame b o1
        let (r, o3) ← frame b o2
        let some c := String.fromUTF8? c | throw "transcript: a channel name is not UTF-8"
        go fuel o3 (acc.push (.asked c body r))
      | 2 =>
        let (d, o) ← frame b (off + 1)
        go fuel o (acc.push (.drew d))
      | t => throw s!"transcript: unknown record tag {t} at byte {off}"
  go (b.size + 1) 0 #[]

end FlockAudit.Transcript
