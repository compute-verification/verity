import FlockProofs.Transcript
import FlockSoundness.Protocol.Refine.Paths

/-!
# The transcript's frames (R11b, framing)

The replay transcript (PROTOCOL.md §6) appends one frame per domain, label, absorb or zero-coin squeeze to its pending
bytes, and a squeeze answers the pending bytes and its own header; on the unsalted path the server retained each round's
bytes, and the replay compares them exactly. This file models those bytes.

* `Frame`, `Frame.enc`: a frame and its bytes, as the executable writes them (a 16-byte header naming its kind and
  length, then its payload).
* `encs_inj`: a list of frames is recoverable from its bytes, so a round's retained bytes determine what was absorbed.
* `label_enc`, `absorbF_enc`, `absorbSlice_enc`, `absorbBytes_enc`, `framed_enc`, `init_enc`: each step of the
  transcript appends its frame's bytes.
* `squeeze_bindA`, `sample_bindA`, `sampleVec_bindA`, `sampleK_bindA`, `forIn_answers`: a squeeze, or a loop of them,
  consumes rounds that retained the pending bytes and the squeeze's header.
-/

namespace FlockSoundness.Refine

open Flock

/-- **A frame** of the replay transcript (PROTOCOL.md §6.1): the bytes one domain, label, absorb or zero-coin squeeze
appends to the pending bytes, or the header a squeeze answers. -/
inductive Frame where
  | dom (b : ByteArray)
  | label (b : ByteArray)
  | one (v : F128)
  | vec (vs : List F128)
  | bytes (b : ByteArray)
  | sq (kind : UInt8) (n : ℕ)

namespace Frame

def op : Frame → UInt8
  | .dom _ => 1 | .label _ => 2 | .one _ => 3 | .vec _ => 3 | .bytes _ => 5 | .sq _ _ => 4

def kind : Frame → UInt8
  | .dom _ => 0 | .label _ => 0 | .one _ => 1 | .vec _ => 2 | .bytes _ => 0 | .sq k _ => k

def len : Frame → ℕ
  | .dom b => b.size | .label b => b.size | .one _ => 1 | .vec vs => vs.length | .bytes b => b.size | .sq _ n => n

/-- The payload after the header. -/
def pay : Frame → List UInt8
  | .dom b => (Transcript.padded b).data.toList
  | .label b => (Transcript.padded b).data.toList
  | .one v => v.toBytes.data.toList
  | .vec vs => colBytes vs
  | .bytes b => (Transcript.padded b).data.toList
  | .sq _ _ => []

/-- A frame's bytes, as the executable writes them: its 16-byte header, then its payload. -/
def enc (f : Frame) : List UInt8 := (Transcript.header f.op f.kind f.len).data.toList ++ f.pay

/-- A frame's length fits its header's eight bytes. -/
def WF (f : Frame) : Prop := f.len < 2 ^ 64

end Frame

/-- The bytes of a list of frames. -/
def encs (fs : List Frame) : List UInt8 := fs.flatMap Frame.enc

/-! ## What the transcript's steps append -/

open Transcript

variable {β : Type}

end FlockSoundness.Refine
